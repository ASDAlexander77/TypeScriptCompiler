Args: tsc.exe --emit=mlir-llvm -debug c:\temp\1.ts 
Load new dialect in Context 
Load new dialect in Context ts
Load new dialect in Context std
Load new dialect in Context math
Load new dialect in Context llvm
Load new dialect in Context async

!! discovering 'ret type' & 'captured vars' for : main

!! variable = a type: !ts.union<!ts.number,!ts.string> op: %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

!! variable = b type: !ts.union<!ts.number,!ts.string,!ts.boolean> op: %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string>

!! variable: b type: !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string,!ts.boolean>

!! variable = a type: !ts.union<!ts.number,!ts.string> op: %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

!! variable = b type: !ts.union<!ts.number,!ts.string,!ts.boolean> op: %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string>

!! variable: b type: !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string,!ts.boolean>

!! reg. func: main type:() -> ()

!! reg. func: main num inputs:0

!! variable = a type: !ts.union<!ts.number,!ts.string> op: %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

!! variable = b type: !ts.union<!ts.number,!ts.string,!ts.boolean> op: %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string>

!! variable: b type: !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string,!ts.boolean>

!! re-process. func: main type:() -> ()

!! re-process. func: main num inputs:0
Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%1) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %6 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %7, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
}


The pattern rewrite doesn't converge after scanning 10 times

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x20055673f00) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Entry'(0x2005574ea60) {
  "ts.Entry"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Entry -> ()' {
Trying to match "`anonymous-namespace'::EntryOpLowering"
    ** Insert  : 'ts.ReturnInternal'(0x2005574dc50)
    ** Erase   : 'ts.Entry'(0x2005574ea60)
"`anonymous-namespace'::EntryOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'ts.ReturnInternal'(0x2005574dc50) {
      "ts.ReturnInternal"() : () -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
^bb1:  // no predecessors
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x200556348f0) {
  %0 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x20055634c10) {
  %1 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055635a70) {
  %2 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.TypeOf'(0x2005558e4c0) {
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.TypeOf -> ()' {
Trying to match "`anonymous-namespace'::TypeOfOpLowering"
    ** Insert  : 'ts.Constant'(0x20055637870)
    ** Replace : 'ts.TypeOf'(0x2005558e4c0)
"`anonymous-namespace'::TypeOfOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Constant'(0x20055637870) {
      %3 = "ts.Constant"() {value = "number"} : () -> !ts.string

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %5 = ts.CreateUnionInstance %2, %4 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %7, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.Exit"() : () -> ()
^bb1:  // no predecessors
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x20055696e30) {
  %5 = "ts.CreateUnionInstance"(%2, %4) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d63d0) {
  "ts.Store"(%5, %0) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x2005558fcc0) {
  %6 = "ts.Load"(%0) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x2005558ec40) {
  %7 = "ts.Cast"(%6) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.number,!ts.string,!ts.boolean>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d78f0) {
  "ts.Store"(%7, %1) : (!ts.union<!ts.number,!ts.string,!ts.boolean>, !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Exit'(0x2005574dfb0) {
  "ts.Exit"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Exit -> ()' {
Trying to match "`anonymous-namespace'::ExitOpLowering"
    ** Insert  : 'std.br'(0x20055675d40)
    ** Erase   : 'ts.Exit'(0x2005574dfb0)
"`anonymous-namespace'::ExitOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'std.br'(0x20055675d40) {
      "std.br"()[^bb1] : () -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %5 = ts.CreateUnionInstance %2, %4 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %7, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  br ^bb1
  "ts.Exit"() : () -> ()
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

!! Processing function: 
main

!! AFTER FUNC DUMP: 
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  br ^bb1
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}
Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  br ^bb1
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  br ^bb1
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}


Trying to match ""
"" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %1 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


The pattern rewrite doesn't converge after scanning 10 times

Insert const at: 
%0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

const to insert: 
%2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
** Insert  : 'ts.Constant'(0x20055636d30)

const to insert: 
%3 = ts.Constant {value = "number"} : !ts.string
** Insert  : 'ts.Constant'(0x20055635a70)

!! AFTER CONST RELOC FUNC DUMP: 
ts.Func @main () -> ()  {
  %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %1 = ts.Constant {value = "number"} : !ts.string
  %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}
Outlined 0 functions built from async.execute operations

//===-------------------------------------------===//
Legalizing operation : 'module'(0x20055674740) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x20055673f00) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055636d30) {
  %0 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055635a70) {
  %1 = "ts.Constant"() {value = "number"} : () -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x200556348f0) {
  %2 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x20055634c10) {
  %3 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x20055696e30) {
  %4 = "ts.CreateUnionInstance"(%0, %1) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d63d0) {
  "ts.Store"(%4, %2) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x2005558fcc0) {
  %5 = "ts.Load"(%2) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x2005558ec40) {
  %6 = "ts.Cast"(%5) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.number,!ts.string,!ts.boolean>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d78f0) {
  "ts.Store"(%6, %3) : (!ts.union<!ts.number,!ts.string,!ts.boolean>, !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.ReturnInternal'(0x2005574dc50) {
  "ts.ReturnInternal"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'module'(0x20055674740) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x20055673f00) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055636d30) {
  %0 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055635a70) {
  %1 = "ts.Constant"() {value = "number"} : () -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x200556348f0) {
  %2 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x20055634c10) {
  %3 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x20055696e30) {
  %4 = "ts.CreateUnionInstance"(%0, %1) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d63d0) {
  "ts.Store"(%4, %2) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x2005558fcc0) {
  %5 = "ts.Load"(%2) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x2005558ec40) {
  %6 = "ts.Cast"(%5) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.number,!ts.string,!ts.boolean>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d78f0) {
  "ts.Store"(%6, %3) : (!ts.union<!ts.number,!ts.string,!ts.boolean>, !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.ReturnInternal'(0x2005574dc50) {
  "ts.ReturnInternal"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556748a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556743d0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672dd0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055672dd0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !async.group
  func private @mlirAsyncRuntimeEmplaceToken(!async.token)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!async.token)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!async.token) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!async.group) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!async.token)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672e80) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673820) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055673820) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!async.token)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!async.token)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!async.token) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!async.group) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!async.token)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055674a00) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055674a00) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!async.token)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!async.token) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!async.group) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!async.token)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556741c0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672a60) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055672a60) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!async.token) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!async.group) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!async.token)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672f30) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672fe0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055672fe0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!async.group) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!async.token)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556738d0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672590) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055672590) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!async.token)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673610) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055673610) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!async.group)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556736c0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055674270) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055674270) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!async.token, !async.group) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673c40) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055674c10) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556727a0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x200556727a0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!async.token, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672640) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x20055672640) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!async.group, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673090) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556726f0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x200556726f0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x200556732a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x20055674cc0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

!! BEFORE DUMP: 
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}
Ignoring pattern 'unrealized_conversion_cast' because it is impossible to match or cannot lead to legal IR (by cost model)

//===-------------------------------------------===//
Legalizing operation : 'module'(0x20055674740) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x20055673f00) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpLowering"
    ** Insert  : 'func'(0x2005567c250)
    ** Erase   : 'ts.Func'(0x20055673f00)
"`anonymous-namespace'::FuncOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x2005567c250) {
      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
        ** Insert  : 'llvm.func'(0x2005567c9e0)
        ** Erase   : 'func'(0x2005567c250)
"`anonymous-namespace'::FuncOpConversion" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.func'(0x2005567c9e0) {
        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @main() {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


    } -> SUCCESS
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @main() {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    ts.Store %6, %3 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055636d30) {
  %0 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Constant -> ()' {
Trying to match "`anonymous-namespace'::ConstantOpLowering"
    ** Insert  : 'std.constant'(0x2005563aed0)
    ** Replace : 'ts.Constant'(0x20055636d30)
"`anonymous-namespace'::ConstantOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'std.constant'(0x2005563aed0) {
      %0 = "std.constant"() {value = 1.000000e+01 : f64} : () -> f64

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'std.constant -> ()' {
Trying to match "`anonymous-namespace'::ConstantOpLowering"
        ** Insert  : 'llvm.mlir.constant'(0x20055638b30)
        ** Replace : 'std.constant'(0x2005563aed0)
"`anonymous-namespace'::ConstantOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x20055638b30) {
          %0 = "llvm.mlir.constant"() {value = 1.000000e+01 : f64} : () -> f64

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %1 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %2 = ts.Constant {value = "number"} : !ts.string
  %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %4 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %5 = ts.CreateUnionInstance %1, %2 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %3 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%3) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %7, %4 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %1 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %2 = ts.Constant {value = "number"} : !ts.string
  %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %4 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %5 = ts.CreateUnionInstance %1, %2 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %3 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%3) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %7, %4 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x20055635a70) {
  %3 = "ts.Constant"() {value = "number"} : () -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Constant -> ()' {
Trying to match "`anonymous-namespace'::ConstantOpLowering"
    ** Insert  : 'llvm.mlir.global'(0x2005567cd50)
    ** Insert  : 'llvm.mlir.addressof'(0x2005563b010)
    ** Insert  : 'llvm.mlir.constant'(0x20055639710)
    ** Insert  : 'llvm.getelementptr'(0x200554e3630)
    ** Replace : 'ts.Constant'(0x20055635a70)
"`anonymous-namespace'::ConstantOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.global'(0x2005567cd50) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.addressof'(0x2005563b010) {
      %3 = "llvm.mlir.addressof"() {global_name = @s_9237349086447201248} : () -> !llvm.ptr<array<7 x i8>>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.constant'(0x20055639710) {
      %4 = "llvm.mlir.constant"() {value = 0 : i64} : () -> i64

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.getelementptr'(0x200554e3630) {
      %5 = "llvm.getelementptr"(%3, %4, %4) : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %1 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %2 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.getelementptr %2[%3, %3] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %5 = ts.Constant {value = "number"} : !ts.string
  %6 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %7 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %8 = ts.CreateUnionInstance %1, %5 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %8, %6 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %9 = ts.Load(%6) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %10 = ts.Cast %9 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %10, %7 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x200556348f0) {
  %7 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.number,!ts.string> is captured: 0
    ** Insert  : 'llvm.mlir.constant'(0x20055638950)
    ** Insert  : 'llvm.alloca'(0x2005558fb40)
    ** Replace : 'ts.Variable'(0x200556348f0)
"`anonymous-namespace'::VariableOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.constant'(0x20055638950) {
      %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.alloca'(0x2005558fb40) {
      %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %3 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %4 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %5 = llvm.mlir.constant(0 : i64) : i64
  %6 = llvm.getelementptr %4[%5, %5] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %7 = ts.Constant {value = "number"} : !ts.string
  %8 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %9 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %10 = ts.CreateUnionInstance %3, %7 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %10, %8 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %11 = ts.Load(%8) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %12 = ts.Cast %11 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %12, %9 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x20055634c10) {
  %10 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.number,!ts.string,!ts.boolean> is captured: 0
    ** Insert  : 'llvm.mlir.constant'(0x20055639490)
    ** Insert  : 'llvm.alloca'(0x2005558e640)
    ** Replace : 'ts.Variable'(0x20055634c10)
"`anonymous-namespace'::VariableOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.constant'(0x20055639490) {
      %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.alloca'(0x2005558e640) {
      %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %5 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %6 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %7 = llvm.mlir.constant(0 : i64) : i64
  %8 = llvm.getelementptr %6[%7, %7] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %9 = ts.Constant {value = "number"} : !ts.string
  %10 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %11 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %12 = ts.CreateUnionInstance %5, %9 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %12, %10 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %13 = ts.Load(%10) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %14 = ts.Cast %13 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %14, %11 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x20055696e30) {
  %13 = "ts.CreateUnionInstance"(%6, %10) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.CreateUnionInstance -> ()' {
Trying to match "`anonymous-namespace'::CreateUnionInstanceOpLowering"
    ** Insert  : 'llvm.mlir.undef'(0x20055639e90)
    ** Insert  : 'llvm.insertvalue'(0x200556992f0)
    ** Insert  : 'llvm.insertvalue'(0x2005568d890)
    ** Insert  : 'ts.Variable'(0x2005558f240)
    ** Insert  : 'ts.Variable'(0x2005563a1b0)
    ** Insert  : 'ts.MemoryCopy'(0x200555d64a0)
    ** Insert  : 'ts.Load'(0x20055590ec0)
    ** Replace : 'ts.CreateUnionInstance'(0x20055696e30)
"`anonymous-namespace'::CreateUnionInstanceOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.undef'(0x20055639e90) {
      %13 = "llvm.mlir.undef"() : () -> !llvm.struct<(ptr<i8>, f64)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.insertvalue'(0x200556992f0) {
      %14 = "llvm.insertvalue"(%13, %10) {position = [0 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, !ts.string) -> !llvm.struct<(ptr<i8>, f64)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.insertvalue'(0x2005568d890) {
      %15 = "llvm.insertvalue"(%14, %6) {position = [1 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, !ts.number) -> !llvm.struct<(ptr<i8>, f64)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Variable'(0x2005558f240) {
      %16 = "ts.Variable"(%15) {captured = false} : (!llvm.struct<(ptr<i8>, f64)>) -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !llvm.struct<(ptr<i8>, f64)> is captured: 0
        ** Insert  : 'llvm.mlir.constant'(0x20055638c70)
        ** Insert  : 'llvm.alloca'(0x2005558e100)
        ** Insert  : 'llvm.store'(0x200555d8fb0)
        ** Replace : 'ts.Variable'(0x2005558f240)
"`anonymous-namespace'::VariableOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x20055638c70) {
          %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.alloca'(0x2005558e100) {
          %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.store'(0x200555d8fb0) {
          "llvm.store"(%17, %1) : (!llvm.struct<(ptr<i8>, f64)>, !llvm.ptr<struct<(ptr<i8>, f64)>>) -> ()

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %7 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %8 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %9 = llvm.mlir.constant(0 : i64) : i64
  %10 = llvm.getelementptr %8[%9, %9] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %11 = ts.Constant {value = "number"} : !ts.string
  %12 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %13 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %14 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %15 = llvm.insertvalue %11, %14[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %16 = llvm.insertvalue %7, %15[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %16, %1 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %17 = ts.Variable(%16) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  "ts.MemoryCopy"(%18, %17) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %19 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %20 = ts.CreateUnionInstance %7, %11 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %20, %12 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = ts.Load(%12) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %22 = ts.Cast %21 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %22, %13 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Variable'(0x2005563a1b0) {
      %19 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.number,!ts.string> is captured: 0
        ** Insert  : 'llvm.mlir.constant'(0x2005563a250)
        ** Insert  : 'llvm.alloca'(0x200555908c0)
        ** Replace : 'ts.Variable'(0x2005563a1b0)
"`anonymous-namespace'::VariableOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x2005563a250) {
          %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.alloca'(0x200555908c0) {
          %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %21 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %22 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %22, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %23 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %24 = ts.Cast %23 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %24, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.MemoryCopy'(0x200555d64a0) {
      "ts.MemoryCopy"(%21, %20) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.MemoryCopy -> ()' {
Trying to match "`anonymous-namespace'::MemoryCopyOpLowering"
        ** Insert  : 'llvm.func'(0x2005567d220)
        ** Insert  : 'llvm.bitcast'(0x2005558e940)
        ** Insert  : 'llvm.bitcast'(0x2005558ef40)
        ** Insert  : 'ts.SizeOf'(0x20055638f90)
        ** Insert  : 'ts.SizeOf'(0x20055638db0)
        ** Insert  : 'llvm.icmp'(0x2005568d0b0)
        ** Insert  : 'llvm.select'(0x200554ec930)
        ** Insert  : 'llvm.mlir.constant'(0x200556395d0)
        ** Insert  : 'llvm.call'(0x200555b5e40)
        ** Erase   : 'ts.MemoryCopy'(0x200555d64a0)
"`anonymous-namespace'::MemoryCopyOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.func'(0x2005567d220) {
        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.bitcast'(0x2005558e940) {
          %22 = "llvm.bitcast"(%1) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<i8>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.bitcast'(0x2005558ef40) {
          %23 = "llvm.bitcast"(%3) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> !llvm.ptr<i8>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'ts.SizeOf'(0x20055638f90) {
          %24 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64

          * Fold {
          } -> FAILURE : unable to fold

          * Pattern : 'ts.SizeOf -> ()' {
Trying to match "`anonymous-namespace'::SizeOfOpLowering"
            ** Insert  : 'llvm.mlir.null'(0x20055639fd0)

!! size of - storage type: [!llvm.struct<(ptr<i8>, f64)>] llvm storage type: [!llvm.struct<(ptr<i8>, f64)>] llvm ptr: [!llvm.ptr<struct<(ptr<i8>, f64)>>] value: [%23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>]
            ** Insert  : 'llvm.mlir.constant'(0x20055639d50)
            ** Insert  : 'llvm.getelementptr'(0x2005568c2b0)
            ** Insert  : 'llvm.ptrtoint'(0x2005558f300)
            ** Replace : 'ts.SizeOf'(0x20055638f90)
"`anonymous-namespace'::SizeOfOpLowering" result 1

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.null'(0x20055639fd0) {
              %24 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, f64)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.constant'(0x20055639d50) {
              %25 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.getelementptr'(0x2005568c2b0) {
              %26 = "llvm.getelementptr"(%24, %25) : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.ptrtoint'(0x2005558f300) {
              %27 = "llvm.ptrtoint"(%26) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//
          } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %29 = llvm.icmp "ult" %27, %28 : i64
  %30 = llvm.select %29, %27, %28 : i1, i64
  %31 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %30, %31) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %32 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %33 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %33, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %34 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %35 = ts.Cast %34 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %35, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


        } -> SUCCESS
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'ts.SizeOf'(0x20055638db0) {
          %29 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64

          * Fold {
          } -> FAILURE : unable to fold

          * Pattern : 'ts.SizeOf -> ()' {
Trying to match "`anonymous-namespace'::SizeOfOpLowering"
            ** Insert  : 'llvm.mlir.null'(0x2005563a070)

!! size of - storage type: [!llvm.struct<(ptr<i8>, ptr<i8>)>] llvm storage type: [!llvm.struct<(ptr<i8>, ptr<i8>)>] llvm ptr: [!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>] value: [%28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>]
            ** Insert  : 'llvm.mlir.constant'(0x20055639cb0)
            ** Insert  : 'llvm.getelementptr'(0x2005568adb0)
            ** Insert  : 'llvm.ptrtoint'(0x20055590c80)
            ** Replace : 'ts.SizeOf'(0x20055638db0)
"`anonymous-namespace'::SizeOfOpLowering" result 1

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.null'(0x2005563a070) {
              %29 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.constant'(0x20055639cb0) {
              %30 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.getelementptr'(0x2005568adb0) {
              %31 = "llvm.getelementptr"(%29, %30) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.ptrtoint'(0x20055590c80) {
              %32 = "llvm.ptrtoint"(%31) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//
          } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %33 = llvm.icmp "ult" %27, %32 : i64
  %34 = llvm.select %33, %27, %32 : i1, i64
  %35 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %34, %35) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %36 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %37 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %37, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %38 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %39 = ts.Cast %38 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %39, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


        } -> SUCCESS
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.icmp'(0x2005568d0b0) {
          %34 = "llvm.icmp"(%28, %33) {predicate = 6 : i64} : (i64, i64) -> i1

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.select'(0x200554ec930) {
          %35 = "llvm.select"(%34, %28, %33) : (i1, i64, i64) -> i64

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x200556395d0) {
          %36 = "llvm.mlir.constant"() {value = false} : () -> i1

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.call'(0x200555b5e40) {
          "llvm.call"(%22, %23, %35, %36) {callee = @llvm.memcpy.p0i8.p0i8.i64} : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %33 = llvm.icmp "ult" %27, %32 : i64
  %34 = llvm.select %33, %27, %32 : i1, i64
  %35 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %34, %35) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %36 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %37 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %37, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %38 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %39 = ts.Cast %38 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %39, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Load'(0x20055590ec0) {
      %37 = "ts.Load"(%21) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Load -> ()' {
Trying to match "`anonymous-namespace'::LoadOpLowering"
        ** Insert  : 'llvm.load'(0x20055590d40)
        ** Replace : 'ts.Load'(0x20055590ec0)
"`anonymous-namespace'::LoadOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.load'(0x20055590d40) {
          %37 = "llvm.load"(%1) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.struct<(ptr<i8>, ptr<i8>)>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %33 = llvm.icmp "ult" %27, %32 : i64
  %34 = llvm.select %33, %27, %32 : i1, i64
  %35 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %34, %35) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %36 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %37 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %38 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %38, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %39 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %40 = ts.Cast %39 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %40, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %33 = llvm.icmp "ult" %27, %32 : i64
  %34 = llvm.select %33, %27, %32 : i1, i64
  %35 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %34, %35) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %36 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %37 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %38 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %38, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %39 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %40 = ts.Cast %39 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %40, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d63d0) {
  "ts.Store"(%39, %15) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Store -> ()' {
Trying to match "`anonymous-namespace'::StoreOpLowering"
    ** Insert  : 'llvm.store'(0x200555d85f0)
    ** Replace : 'ts.Store'(0x200555d63d0)
"`anonymous-namespace'::StoreOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.store'(0x200555d85f0) {
      "llvm.store"(%37, %7) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %33 = llvm.icmp "ult" %27, %32 : i64
  %34 = llvm.select %33, %27, %32 : i1, i64
  %35 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %34, %35) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %36 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %37 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %38 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %36, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %38, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %39 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %40 = ts.Cast %39 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %40, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x2005558fcc0) {
  %40 = "ts.Load"(%15) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Load -> ()' {
Trying to match "`anonymous-namespace'::LoadOpLowering"
    ** Insert  : 'llvm.load'(0x2005558f0c0)
    ** Replace : 'ts.Load'(0x2005558fcc0)
"`anonymous-namespace'::LoadOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.load'(0x2005558f0c0) {
      %40 = "llvm.load"(%7) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.struct<(ptr<i8>, ptr<i8>)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %10 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.getelementptr %10[%11, %11] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %13 = ts.Constant {value = "number"} : !ts.string
  %14 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %17 = llvm.insertvalue %13, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %9, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %18, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %19 = ts.Variable(%18) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %22 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = llvm.mlir.constant(1 : i64) : i64
  %25 = llvm.getelementptr %23[%24] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.ptrtoint %25 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %27 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %33 = llvm.icmp "ult" %27, %32 : i64
  %34 = llvm.select %33, %27, %32 : i1, i64
  %35 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%21, %22, %34, %35) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%20, %19) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %36 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %37 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %38 = ts.CreateUnionInstance %9, %13 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %36, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %38, %14 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %39 = llvm.load %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %40 = ts.Load(%14) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %41, %15 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x2005558ec40) {
  %42 = "ts.Cast"(%41) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.number,!ts.string,!ts.boolean>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Cast -> ()' {
Trying to match "`anonymous-namespace'::CastOpLowering"
    ** Insert  : 'ts.Variable'(0x2005558fd80)
    ** Insert  : 'ts.Variable'(0x2005563a110)
    ** Insert  : 'ts.MemoryCopy'(0x200555d8860)
    ** Insert  : 'ts.Load'(0x20055590980)
    ** Replace : 'ts.Cast'(0x2005558ec40)
"`anonymous-namespace'::CastOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Variable'(0x2005558fd80) {
      %42 = "ts.Variable"(%40) {captured = false} : (!llvm.struct<(ptr<i8>, ptr<i8>)>) -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !llvm.struct<(ptr<i8>, ptr<i8>)> is captured: 0
        ** Insert  : 'llvm.mlir.constant'(0x200556389f0)
        ** Insert  : 'llvm.alloca'(0x2005558e7c0)
        ** Insert  : 'llvm.store'(0x200555d8c70)
        ** Replace : 'ts.Variable'(0x2005558fd80)
"`anonymous-namespace'::VariableOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x200556389f0) {
          %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.alloca'(0x2005558e7c0) {
          %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.store'(0x200555d8c70) {
          "llvm.store"(%42, %1) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %10 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %11 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %12 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %13 = llvm.mlir.constant(0 : i64) : i64
  %14 = llvm.getelementptr %12[%13, %13] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %15 = ts.Constant {value = "number"} : !ts.string
  %16 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %17 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %18 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %19 = llvm.insertvalue %15, %18[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %20 = llvm.insertvalue %11, %19[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %20, %5 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %21 = ts.Variable(%20) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %22 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %23 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %24 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %25 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %26 = llvm.mlir.constant(1 : i64) : i64
  %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.ptrtoint %27 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %29 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %30 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %31 = llvm.mlir.constant(1 : i64) : i64
  %32 = llvm.getelementptr %30[%31] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.ptrtoint %32 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %34 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %35 = llvm.icmp "ult" %29, %34 : i64
  %36 = llvm.select %35, %29, %34 : i1, i64
  %37 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%23, %24, %36, %37) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%22, %21) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %38 = llvm.load %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %39 = ts.Load(%22) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %40 = ts.CreateUnionInstance %11, %15 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %38, %9 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %40, %16 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %41 = llvm.load %9 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %42 = ts.Load(%16) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %41, %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %43 = ts.Variable(%41) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.MemoryCopy"(%44, %43) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %45 = ts.Load(%44) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %46 = ts.Cast %42 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %46, %17 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Variable'(0x2005563a110) {
      %45 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.number,!ts.string,!ts.boolean> is captured: 0
        ** Insert  : 'llvm.mlir.constant'(0x2005563ab10)
        ** Insert  : 'llvm.alloca'(0x2005558e1c0)
        ** Replace : 'ts.Variable'(0x2005563a110)
"`anonymous-namespace'::VariableOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x2005563ab10) {
          %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.alloca'(0x2005558e1c0) {
          %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %47 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %48 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %48, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.MemoryCopy'(0x200555d8860) {
      "ts.MemoryCopy"(%47, %46) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.MemoryCopy -> ()' {
Trying to match "`anonymous-namespace'::MemoryCopyOpLowering"
        ** Insert  : 'llvm.bitcast'(0x2005558e4c0)
        ** Insert  : 'llvm.bitcast'(0x20055590140)
        ** Insert  : 'ts.SizeOf'(0x20055638e50)
        ** Insert  : 'ts.SizeOf'(0x2005563a2f0)
        ** Insert  : 'llvm.icmp'(0x2005568acd0)
        ** Insert  : 'llvm.select'(0x200554ee430)
        ** Insert  : 'llvm.mlir.constant'(0x20055638a90)
        ** Insert  : 'llvm.call'(0x200555b6e30)
        ** Erase   : 'ts.MemoryCopy'(0x200555d8860)
"`anonymous-namespace'::MemoryCopyOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.bitcast'(0x2005558e4c0) {
          %48 = "llvm.bitcast"(%1) : (!llvm.ptr<struct<(ptr<i8>, i1)>>) -> !llvm.ptr<i8>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.bitcast'(0x20055590140) {
          %49 = "llvm.bitcast"(%3) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<i8>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'ts.SizeOf'(0x20055638e50) {
          %50 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64

          * Fold {
          } -> FAILURE : unable to fold

          * Pattern : 'ts.SizeOf -> ()' {
Trying to match "`anonymous-namespace'::SizeOfOpLowering"
            ** Insert  : 'llvm.mlir.null'(0x2005563a6b0)

!! size of - storage type: [!llvm.struct<(ptr<i8>, ptr<i8>)>] llvm storage type: [!llvm.struct<(ptr<i8>, ptr<i8>)>] llvm ptr: [!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>] value: [%49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>]
            ** Insert  : 'llvm.mlir.constant'(0x2005563a430)
            ** Insert  : 'llvm.getelementptr'(0x2005568b830)
            ** Insert  : 'llvm.ptrtoint'(0x20055590380)
            ** Replace : 'ts.SizeOf'(0x20055638e50)
"`anonymous-namespace'::SizeOfOpLowering" result 1

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.null'(0x2005563a6b0) {
              %50 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.constant'(0x2005563a430) {
              %51 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.getelementptr'(0x2005568b830) {
              %52 = "llvm.getelementptr"(%50, %51) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.ptrtoint'(0x20055590380) {
              %53 = "llvm.ptrtoint"(%52) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//
          } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %55 = llvm.icmp "ult" %53, %54 : i64
  %56 = llvm.select %55, %53, %54 : i1, i64
  %57 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %56, %57) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %58 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %59 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %59, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


        } -> SUCCESS
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'ts.SizeOf'(0x2005563a2f0) {
          %55 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64

          * Fold {
          } -> FAILURE : unable to fold

          * Pattern : 'ts.SizeOf -> ()' {
Trying to match "`anonymous-namespace'::SizeOfOpLowering"
            ** Insert  : 'llvm.mlir.null'(0x2005563a750)

!! size of - storage type: [!llvm.struct<(ptr<i8>, i1)>] llvm storage type: [!llvm.struct<(ptr<i8>, i1)>] llvm ptr: [!llvm.ptr<struct<(ptr<i8>, i1)>>] value: [%54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>]
            ** Insert  : 'llvm.mlir.constant'(0x20055639350)
            ** Insert  : 'llvm.getelementptr'(0x2005568be50)
            ** Insert  : 'llvm.ptrtoint'(0x20055590800)
            ** Replace : 'ts.SizeOf'(0x2005563a2f0)
"`anonymous-namespace'::SizeOfOpLowering" result 1

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.null'(0x2005563a750) {
              %55 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, i1)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.constant'(0x20055639350) {
              %56 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.getelementptr'(0x2005568be50) {
              %57 = "llvm.getelementptr"(%55, %56) : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.ptrtoint'(0x20055590800) {
              %58 = "llvm.ptrtoint"(%57) : (!llvm.ptr<struct<(ptr<i8>, i1)>>) -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//
          } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %55 = llvm.mlir.constant(1 : i64) : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %59 = llvm.icmp "ult" %53, %58 : i64
  %60 = llvm.select %59, %53, %58 : i1, i64
  %61 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %62 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %63 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %63, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


        } -> SUCCESS
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.icmp'(0x2005568acd0) {
          %60 = "llvm.icmp"(%54, %59) {predicate = 6 : i64} : (i64, i64) -> i1

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.select'(0x200554ee430) {
          %61 = "llvm.select"(%60, %54, %59) : (i1, i64, i64) -> i64

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x20055638a90) {
          %62 = "llvm.mlir.constant"() {value = false} : () -> i1

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.call'(0x200555b6e30) {
          "llvm.call"(%48, %49, %61, %62) {callee = @llvm.memcpy.p0i8.p0i8.i64} : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %55 = llvm.mlir.constant(1 : i64) : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %59 = llvm.icmp "ult" %53, %58 : i64
  %60 = llvm.select %59, %53, %58 : i1, i64
  %61 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %62 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %63 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %63, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Load'(0x20055590980) {
      %63 = "ts.Load"(%47) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>) -> !ts.union<!ts.number,!ts.string,!ts.boolean>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Load -> ()' {
Trying to match "`anonymous-namespace'::LoadOpLowering"
        ** Insert  : 'llvm.load'(0x2005558f6c0)
        ** Replace : 'ts.Load'(0x20055590980)
"`anonymous-namespace'::LoadOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.load'(0x2005558f6c0) {
          %63 = "llvm.load"(%1) : (!llvm.ptr<struct<(ptr<i8>, i1)>>) -> !llvm.struct<(ptr<i8>, i1)>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %55 = llvm.mlir.constant(1 : i64) : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %59 = llvm.icmp "ult" %53, %58 : i64
  %60 = llvm.select %59, %53, %58 : i1, i64
  %61 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %55 = llvm.mlir.constant(1 : i64) : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %59 = llvm.icmp "ult" %53, %58 : i64
  %60 = llvm.select %59, %53, %58 : i1, i64
  %61 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x200555d78f0) {
  "ts.Store"(%65, %20) : (!ts.union<!ts.number,!ts.string,!ts.boolean>, !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Store -> ()' {
Trying to match "`anonymous-namespace'::StoreOpLowering"
    ** Insert  : 'llvm.store'(0x200555d9150)
    ** Replace : 'ts.Store'(0x200555d78f0)
"`anonymous-namespace'::StoreOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.store'(0x200555d9150) {
      "llvm.store"(%63, %9) : (!llvm.struct<(ptr<i8>, i1)>, !llvm.ptr<struct<(ptr<i8>, i1)>>) -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %55 = llvm.mlir.constant(1 : i64) : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %59 = llvm.icmp "ult" %53, %58 : i64
  %60 = llvm.select %59, %53, %58 : i1, i64
  %61 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.ReturnInternal'(0x2005574dc50) {
  "ts.ReturnInternal"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.ReturnInternal -> ()' {
Trying to match "`anonymous-namespace'::ReturnInternalOpLowering"
    ** Insert  : 'llvm.return'(0x2005574e700)
    ** Replace : 'ts.ReturnInternal'(0x2005574dc50)
"`anonymous-namespace'::ReturnInternalOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.return'(0x2005574e700) {
      "llvm.return"() : () -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %15 = llvm.mlir.constant(0 : i64) : i64
  %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %17 = ts.Constant {value = "number"} : !ts.string
  %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %28 = llvm.mlir.constant(1 : i64) : i64
  %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %33 = llvm.mlir.constant(1 : i64) : i64
  %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %37 = llvm.icmp "ult" %31, %36 : i64
  %38 = llvm.select %37, %31, %36 : i1, i64
  %39 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %50 = llvm.mlir.constant(1 : i64) : i64
  %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %55 = llvm.mlir.constant(1 : i64) : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %59 = llvm.icmp "ult" %53, %58 : i64
  %60 = llvm.select %59, %53, %58 : i1, i64
  %61 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  llvm.return
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556748a0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567d380)
    ** Erase   : 'func'(0x200556748a0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567d380) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556743d0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567ba10)
    ** Erase   : 'func'(0x200556743d0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567ba10) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672dd0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567b800)
    ** Erase   : 'func'(0x20055672dd0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567b800) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672e80) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567d4e0)
    ** Erase   : 'func'(0x20055672e80)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567d4e0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673820) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567ceb0)
    ** Erase   : 'func'(0x20055673820)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567ceb0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055674a00) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567c7d0)
    ** Erase   : 'func'(0x20055674a00)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567c7d0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556741c0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567bb70)
    ** Erase   : 'func'(0x200556741c0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567bb70) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672a60) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567b5f0)
    ** Erase   : 'func'(0x20055672a60)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567b5f0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672f30) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567adb0)
    ** Erase   : 'func'(0x20055672f30)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567adb0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672fe0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567bc20)
    ** Erase   : 'func'(0x20055672fe0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567bc20) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556738d0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567bee0)
    ** Erase   : 'func'(0x200556738d0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567bee0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672590) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567ca90)
    ** Erase   : 'func'(0x20055672590)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567ca90) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673610) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567afc0)
    ** Erase   : 'func'(0x20055673610)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567afc0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556736c0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567b070)
    ** Erase   : 'func'(0x200556736c0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567b070) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055674270) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567d0c0)
    ** Erase   : 'func'(0x20055674270)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567d0c0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673c40) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567af10)
    ** Erase   : 'func'(0x20055673c40)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567af10) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055674c10) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567b960)
    ** Erase   : 'func'(0x20055674c10)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567b960) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556727a0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567ce00)
    ** Erase   : 'func'(0x200556727a0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567ce00) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055672640) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567cb40)
    ** Erase   : 'func'(0x20055672640)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567cb40) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  llvm.func @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x20055673090) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567bd80)
    ** Erase   : 'func'(0x20055673090)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567bd80) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  llvm.func @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x200556726f0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x2005567aa40)
    ** Erase   : 'func'(0x200556726f0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x2005567aa40) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %cst = constant 1.000000e+01 : f64
    %13 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %14 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %15 = llvm.mlir.constant(0 : i64) : i64
    %16 = llvm.getelementptr %14[%15, %15] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %17 = ts.Constant {value = "number"} : !ts.string
    %18 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %20 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %21 = llvm.insertvalue %17, %20[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %22 = llvm.insertvalue %13, %21[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %22, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %23 = ts.Variable(%22) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
    %24 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %25 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %26 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %27 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %28 = llvm.mlir.constant(1 : i64) : i64
    %29 = llvm.getelementptr %27[%28] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %30 = llvm.ptrtoint %29 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %31 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
    %32 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.mlir.constant(1 : i64) : i64
    %34 = llvm.getelementptr %32[%33] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %35 = llvm.ptrtoint %34 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %36 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %37 = llvm.icmp "ult" %31, %36 : i64
    %38 = llvm.select %37, %31, %36 : i1, i64
    %39 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%25, %26, %38, %39) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%24, %23) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
    %40 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %41 = ts.Load(%24) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %42 = ts.CreateUnionInstance %13, %17 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    llvm.store %40, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %42, %18 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %43 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %44 = ts.Load(%18) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    llvm.store %43, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %45 = ts.Variable(%43) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
    %46 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    %47 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %48 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %49 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %50 = llvm.mlir.constant(1 : i64) : i64
    %51 = llvm.getelementptr %49[%50] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %52 = llvm.ptrtoint %51 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %53 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
    %54 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %55 = llvm.mlir.constant(1 : i64) : i64
    %56 = llvm.getelementptr %54[%55] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %57 = llvm.ptrtoint %56 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %58 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
    %59 = llvm.icmp "ult" %53, %58 : i64
    %60 = llvm.select %59, %53, %58 : i1, i64
    %61 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%47, %48, %60, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    "ts.MemoryCopy"(%46, %45) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
    %62 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %63 = ts.Load(%46) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
    %64 = ts.Cast %44 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
    llvm.store %62, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    ts.Store %64, %19 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
    llvm.return
    ts.ReturnInternal
  }
  func @main()
  ts.Func @main () -> ()  {
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32)
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>)
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64
  llvm.func @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  func private @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>)
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x200556732a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x20055674cc0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

!! SourceMaterialization: loc:[ loc("c:\\temp\\1.ts":5:9) ] result: [ !ts.number ]

!! SourceMaterialization value: [ %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64 ]
** Insert  : 'ts.DialectCast'(0x2005558fe40)

//===-------------------------------------------===//
Legalizing operation : 'ts.DialectCast'(0x2005558fe40) {
  %14 = "ts.DialectCast"(%12) : (f64) -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.DialectCast -> ()' {
Trying to match "`anonymous-namespace'::DialectCastOpLowering"
    ** Replace : 'ts.DialectCast'(0x2005558fe40)
"`anonymous-namespace'::DialectCastOpLowering" result 1
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.DialectCast %12 : f64 to !ts.number
  %14 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %15 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %16 = llvm.mlir.constant(0 : i64) : i64
  %17 = llvm.getelementptr %15[%16, %16] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %18 = ts.Constant {value = "number"} : !ts.string
  %19 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %21 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %22 = llvm.insertvalue %18, %21[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %23 = llvm.insertvalue %14, %22[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %23, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %24 = ts.Variable(%23) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %25 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %26 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %27 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %29 = llvm.mlir.constant(1 : i64) : i64
  %30 = llvm.getelementptr %28[%29] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %31 = llvm.ptrtoint %30 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %32 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %33 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %34 = llvm.mlir.constant(1 : i64) : i64
  %35 = llvm.getelementptr %33[%34] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %36 = llvm.ptrtoint %35 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %37 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %38 = llvm.icmp "ult" %32, %37 : i64
  %39 = llvm.select %38, %32, %37 : i1, i64
  %40 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%26, %27, %39, %40) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%25, %24) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %41 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %42 = ts.Load(%25) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %43 = ts.CreateUnionInstance %14, %18 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %41, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %43, %19 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %44 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %45 = ts.Load(%19) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %44, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Variable(%44) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %47 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %48 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %49 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %50 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %51 = llvm.mlir.constant(1 : i64) : i64
  %52 = llvm.getelementptr %50[%51] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %53 = llvm.ptrtoint %52 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %54 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %55 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %56 = llvm.mlir.constant(1 : i64) : i64
  %57 = llvm.getelementptr %55[%56] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %58 = llvm.ptrtoint %57 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %59 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %60 = llvm.icmp "ult" %54, %59 : i64
  %61 = llvm.select %60, %54, %59 : i1, i64
  %62 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%48, %49, %61, %62) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%47, %46) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %63 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %64 = ts.Load(%47) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %65 = ts.Cast %45 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  llvm.store %63, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  ts.Store %65, %20 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  llvm.return
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

!! SourceMaterialization: loc:[ loc("c:\\temp\\1.ts":5:5) ] result: [ !ts.string ]

!! SourceMaterialization value: [ %17 = llvm.getelementptr %15[%16, %16] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8> ]
** Insert  : 'ts.DialectCast'(0x2005558fa80)

//===-------------------------------------------===//
Legalizing operation : 'ts.DialectCast'(0x2005558fa80) {
  %19 = "ts.DialectCast"(%18) : (!llvm.ptr<i8>) -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.DialectCast -> ()' {
Trying to match "`anonymous-namespace'::DialectCastOpLowering"
    ** Replace : 'ts.DialectCast'(0x2005558fa80)
"`anonymous-namespace'::DialectCastOpLowering" result 1
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %8 = llvm.mlir.constant(1 : i32) : i32
  %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %10 = llvm.mlir.constant(1 : i32) : i32
  %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %13 = ts.DialectCast %12 : f64 to !ts.number
  %14 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %15 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %16 = llvm.mlir.constant(0 : i64) : i64
  %17 = llvm.getelementptr %15[%16, %16] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %18 = ts.DialectCast %17 : !llvm.ptr<i8> to !ts.string
  %19 = ts.Constant {value = "number"} : !ts.string
  %20 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %21 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %22 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %23 = llvm.insertvalue %19, %22[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %24 = llvm.insertvalue %14, %23[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %24, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %25 = ts.Variable(%24) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %26 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %27 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %28 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %29 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %30 = llvm.mlir.constant(1 : i64) : i64
  %31 = llvm.getelementptr %29[%30] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %32 = llvm.ptrtoint %31 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %33 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %34 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %35 = llvm.mlir.constant(1 : i64) : i64
  %36 = llvm.getelementptr %34[%35] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %37 = llvm.ptrtoint %36 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %38 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %39 = llvm.icmp "ult" %33, %38 : i64
  %40 = llvm.select %39, %33, %38 : i1, i64
  %41 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%27, %28, %40, %41) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%26, %25) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %42 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %43 = ts.Load(%26) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %44 = ts.CreateUnionInstance %14, %19 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %42, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %44, %20 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %45 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %46 = ts.Load(%20) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  llvm.store %45, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %47 = ts.Variable(%45) {false} : !llvm.struct<(ptr<i8>, ptr<i8>)> -> !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>
  %48 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  %49 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
  %50 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %51 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %52 = llvm.mlir.constant(1 : i64) : i64
  %53 = llvm.getelementptr %51[%52] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %54 = llvm.ptrtoint %53 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %55 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %56 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %57 = llvm.mlir.constant(1 : i64) : i64
  %58 = llvm.getelementptr %56[%57] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
  %59 = llvm.ptrtoint %58 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
  %60 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, i1)>} : () -> i64
  %61 = llvm.icmp "ult" %55, %60 : i64
  %62 = llvm.select %61, %55, %60 : i1, i64
  %63 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%49, %50, %62, %63) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%48, %47) : (!ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>, !ts.ref<!llvm.struct<(ptr<i8>, ptr<i8>)>>) -> ()
  %64 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  %65 = ts.Load(%48) : !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>> -> !ts.union<!ts.number,!ts.string,!ts.boolean>
  %66 = ts.Cast %46 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.number,!ts.string,!ts.boolean>
  llvm.store %64, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
  ts.Store %66, %21 : !ts.union<!ts.number,!ts.string,!ts.boolean> -> !ts.ref<!ts.union<!ts.number,!ts.string,!ts.boolean>>
  llvm.return
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'module'(0x20055674740) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567d220) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.global'(0x2005567cd50) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567c9e0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x2005563ab10) {
  %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x2005558e1c0) {
  %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x200556389f0) {
  %2 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x2005558e7c0) {
  %3 = "llvm.alloca"(%2) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x2005563a250) {
  %4 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x200555908c0) {
  %5 = "llvm.alloca"(%4) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055638c70) {
  %6 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x2005558e100) {
  %7 = "llvm.alloca"(%6) : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055639490) {
  %8 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x2005558e640) {
  %9 = "llvm.alloca"(%8) : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055638950) {
  %10 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x2005558fb40) {
  %11 = "llvm.alloca"(%10) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055638b30) {
  %12 = "llvm.mlir.constant"() {value = 1.000000e+01 : f64} : () -> f64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.addressof'(0x2005563b010) {
  %13 = "llvm.mlir.addressof"() {global_name = @s_9237349086447201248} : () -> !llvm.ptr<array<7 x i8>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055639710) {
  %14 = "llvm.mlir.constant"() {value = 0 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x200554e3630) {
  %15 = "llvm.getelementptr"(%13, %14, %14) : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.undef'(0x20055639e90) {
  %16 = "llvm.mlir.undef"() : () -> !llvm.struct<(ptr<i8>, f64)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.insertvalue'(0x200556992f0) {
  %17 = "llvm.insertvalue"(%16, %15) {position = [0 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, !llvm.ptr<i8>) -> !llvm.struct<(ptr<i8>, f64)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.insertvalue'(0x2005568d890) {
  %18 = "llvm.insertvalue"(%17, %12) {position = [1 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, f64) -> !llvm.struct<(ptr<i8>, f64)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x200555d8fb0) {
  "llvm.store"(%18, %7) : (!llvm.struct<(ptr<i8>, f64)>, !llvm.ptr<struct<(ptr<i8>, f64)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.bitcast'(0x2005558e940) {
  %19 = "llvm.bitcast"(%5) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.bitcast'(0x2005558ef40) {
  %20 = "llvm.bitcast"(%7) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.null'(0x20055639fd0) {
  %21 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, f64)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055639d50) {
  %22 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x2005568c2b0) {
  %23 = "llvm.getelementptr"(%21, %22) : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.ptrtoint'(0x2005558f300) {
  %24 = "llvm.ptrtoint"(%23) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.null'(0x2005563a070) {
  %25 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055639cb0) {
  %26 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x2005568adb0) {
  %27 = "llvm.getelementptr"(%25, %26) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.ptrtoint'(0x20055590c80) {
  %28 = "llvm.ptrtoint"(%27) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.icmp'(0x2005568d0b0) {
  %29 = "llvm.icmp"(%24, %28) {predicate = 6 : i64} : (i64, i64) -> i1

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.select'(0x200554ec930) {
  %30 = "llvm.select"(%29, %24, %28) : (i1, i64, i64) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x200556395d0) {
  %31 = "llvm.mlir.constant"() {value = false} : () -> i1

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.call'(0x200555b5e40) {
  "llvm.call"(%19, %20, %30, %31) {callee = @llvm.memcpy.p0i8.p0i8.i64} : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.load'(0x20055590d40) {
  %32 = "llvm.load"(%5) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.struct<(ptr<i8>, ptr<i8>)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x200555d85f0) {
  "llvm.store"(%32, %11) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.load'(0x2005558f0c0) {
  %33 = "llvm.load"(%11) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.struct<(ptr<i8>, ptr<i8>)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x200555d8c70) {
  "llvm.store"(%33, %3) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.bitcast'(0x2005558e4c0) {
  %34 = "llvm.bitcast"(%1) : (!llvm.ptr<struct<(ptr<i8>, i1)>>) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.bitcast'(0x20055590140) {
  %35 = "llvm.bitcast"(%3) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.null'(0x2005563a6b0) {
  %36 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x2005563a430) {
  %37 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x2005568b830) {
  %38 = "llvm.getelementptr"(%36, %37) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.ptrtoint'(0x20055590380) {
  %39 = "llvm.ptrtoint"(%38) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.null'(0x2005563a750) {
  %40 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, i1)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055639350) {
  %41 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x2005568be50) {
  %42 = "llvm.getelementptr"(%40, %41) : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.ptrtoint'(0x20055590800) {
  %43 = "llvm.ptrtoint"(%42) : (!llvm.ptr<struct<(ptr<i8>, i1)>>) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.icmp'(0x2005568acd0) {
  %44 = "llvm.icmp"(%39, %43) {predicate = 6 : i64} : (i64, i64) -> i1

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.select'(0x200554ee430) {
  %45 = "llvm.select"(%44, %39, %43) : (i1, i64, i64) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x20055638a90) {
  %46 = "llvm.mlir.constant"() {value = false} : () -> i1

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.call'(0x200555b6e30) {
  "llvm.call"(%34, %35, %45, %46) {callee = @llvm.memcpy.p0i8.p0i8.i64} : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.load'(0x2005558f6c0) {
  %47 = "llvm.load"(%1) : (!llvm.ptr<struct<(ptr<i8>, i1)>>) -> !llvm.struct<(ptr<i8>, i1)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x200555d9150) {
  "llvm.store"(%47, %9) : (!llvm.struct<(ptr<i8>, i1)>, !llvm.ptr<struct<(ptr<i8>, i1)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.return'(0x2005574e700) {
  "llvm.return"() : () -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567d380) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567ba10) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567b800) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567d4e0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567ceb0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567c7d0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567bb70) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567b5f0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567adb0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567bc20) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567bee0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567ca90) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567afc0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567b070) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567d0c0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567af10) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567b960) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567ce00) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567cb40) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567bd80) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x2005567aa40) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x200556732a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x20055674cc0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

!! AFTER DUMP: 
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %13 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.getelementptr %13[%14, %14] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %17 = llvm.insertvalue %15, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %18 = llvm.insertvalue %12, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %18, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %19 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %20 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %21 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %25 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %29 = llvm.icmp "ult" %24, %28 : i64
    %30 = llvm.select %29, %24, %28 : i1, i64
    %31 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%19, %20, %30, %31) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %32 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %32, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %33, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %34 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %35 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %36 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.getelementptr %36[%37] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %39 = llvm.ptrtoint %38 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %40 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %41 = llvm.mlir.constant(1 : i64) : i64
    %42 = llvm.getelementptr %40[%41] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %43 = llvm.ptrtoint %42 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %44 = llvm.icmp "ult" %39, %43 : i64
    %45 = llvm.select %44, %39, %43 : i1, i64
    %46 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%34, %35, %45, %46) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %47 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    llvm.store %47, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    llvm.return
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @free(!llvm.ptr<i8>)
}
** Insert  : 'llvm.func'(0x2005567c670)
** Insert  : 'llvm.call'(0x2005574f120)
module @"c:\\temp\\1.ts"  {
  llvm.func @GC_init()
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    llvm.call @GC_init() : () -> ()
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.alloca %8 x !llvm.struct<(ptr<i8>, i1)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.alloca %10 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %12 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %13 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.getelementptr %13[%14, %14] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %17 = llvm.insertvalue %15, %16[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %18 = llvm.insertvalue %12, %17[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %18, %7 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %19 = llvm.bitcast %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %20 = llvm.bitcast %7 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %21 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %25 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %26 = llvm.mlir.constant(1 : i64) : i64
    %27 = llvm.getelementptr %25[%26] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %28 = llvm.ptrtoint %27 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %29 = llvm.icmp "ult" %24, %28 : i64
    %30 = llvm.select %29, %24, %28 : i1, i64
    %31 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%19, %20, %30, %31) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %32 = llvm.load %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %32, %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %33 = llvm.load %11 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %33, %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %34 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, i1)>> to !llvm.ptr<i8>
    %35 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %36 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %37 = llvm.mlir.constant(1 : i64) : i64
    %38 = llvm.getelementptr %36[%37] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %39 = llvm.ptrtoint %38 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %40 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, i1)>>
    %41 = llvm.mlir.constant(1 : i64) : i64
    %42 = llvm.getelementptr %40[%41] : (!llvm.ptr<struct<(ptr<i8>, i1)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, i1)>>
    %43 = llvm.ptrtoint %42 : !llvm.ptr<struct<(ptr<i8>, i1)>> to i64
    %44 = llvm.icmp "ult" %39, %43 : i64
    %45 = llvm.select %44, %39, %43 : i1, i64
    %46 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%34, %35, %45, %46) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %47 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    llvm.store %47, %9 : !llvm.ptr<struct<(ptr<i8>, i1)>>
    llvm.return
  }
  llvm.func @mlirAsyncRuntimeAddRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr<i8>, i32) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateValue(i32) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeCreateGroup(i64) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeEmplaceToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeEmplaceValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeSetValueError(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsValueError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeIsGroupError(!llvm.ptr<i8>) -> i1 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitAllInGroup(!llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeExecute(!llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeGetValueStorage(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAddTokenToGroup(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i64 attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitTokenAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitValueAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @mlirAsyncRuntimeAwaitAllInGroupAndExecute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<func<void (ptr<i8>)>>) attributes {sym_visibility = "private"}
  llvm.func @GC_malloc(i64) -> !llvm.ptr<i8>
  llvm.func @GC_free(!llvm.ptr<i8>)
}
