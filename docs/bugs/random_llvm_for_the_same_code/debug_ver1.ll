Args: tsc.exe --emit=mlir-llvm -debug c:\temp\1.ts 
Load new dialect in Context 
Load new dialect in Context ts
Load new dialect in Context std
Load new dialect in Context math
Load new dialect in Context llvm
Load new dialect in Context async

!! discovering 'ret type' & 'captured vars' for : main

!! variable = a type: !ts.union<!ts.number,!ts.string> op: %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

!! variable = b type: !ts.union<!ts.boolean,!ts.number,!ts.string> op: %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string>

!! variable: b type: !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.boolean,!ts.number,!ts.string>

!! variable = a type: !ts.union<!ts.number,!ts.string> op: %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

!! variable = b type: !ts.union<!ts.boolean,!ts.number,!ts.string> op: %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string>

!! variable: b type: !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.boolean,!ts.number,!ts.string>

!! reg. func: main type:() -> ()

!! reg. func: main num inputs:0

!! variable = a type: !ts.union<!ts.number,!ts.string> op: %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

!! variable = b type: !ts.union<!ts.boolean,!ts.number,!ts.string> op: %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.number,!ts.string>

!! variable: b type: !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

!! variable: a type: !ts.ref<!ts.union<!ts.number,!ts.string>>

!! Dest type: !ts.union<!ts.boolean,!ts.number,!ts.string>

!! re-process. func: main type:() -> ()

!! re-process. func: main num inputs:0
Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%1) : !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>> -> !ts.union<!ts.boolean,!ts.number,!ts.string>
  %6 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %7, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
}


The pattern rewrite doesn't converge after scanning 10 times

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x1b285d068b0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Entry'(0x1b285d3fb70) {
  "ts.Entry"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Entry -> ()' {
Trying to match "`anonymous-namespace'::EntryOpLowering"
    ** Insert  : 'ts.ReturnInternal'(0x1b285d402c0)
    ** Erase   : 'ts.Entry'(0x1b285d3fb70)
"`anonymous-namespace'::EntryOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'ts.ReturnInternal'(0x1b285d402c0) {
      "ts.ReturnInternal"() : () -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
^bb1:  // no predecessors
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cc9d80) {
  %0 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cca780) {
  %1 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x1b285cca320) {
  %2 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.TypeOf'(0x1b285c12610) {
  %3 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.TypeOf -> ()' {
Trying to match "`anonymous-namespace'::TypeOfOpLowering"
    ** Insert  : 'ts.Constant'(0x1b285ccb0e0)
    ** Replace : 'ts.TypeOf'(0x1b285c12610)
"`anonymous-namespace'::TypeOfOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Constant'(0x1b285ccb0e0) {
      %3 = "ts.Constant"() {value = "number"} : () -> !ts.string

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %5 = ts.CreateUnionInstance %2, %4 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %7, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  "ts.Exit"() : () -> ()
^bb1:  // no predecessors
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x1b285ca2600) {
  %5 = "ts.CreateUnionInstance"(%2, %4) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c67b80) {
  "ts.Store"(%5, %0) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x1b285c13f90) {
  %6 = "ts.Load"(%0) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x1b285c117d0) {
  %7 = "ts.Cast"(%6) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.boolean,!ts.number,!ts.string>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c679e0) {
  "ts.Store"(%7, %1) : (!ts.union<!ts.boolean,!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Exit'(0x1b285d3ef10) {
  "ts.Exit"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Exit -> ()' {
Trying to match "`anonymous-namespace'::ExitOpLowering"
    ** Insert  : 'std.br'(0x1b285d08b10)
    ** Erase   : 'ts.Exit'(0x1b285d3ef10)
"`anonymous-namespace'::ExitOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'std.br'(0x1b285d08b10) {
      "std.br"()[^bb1] : () -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  "ts.Entry"() : () -> ()
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = "ts.TypeOf"(%2) : (!ts.number) -> !ts.string
  %5 = ts.CreateUnionInstance %2, %4 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %7, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  br ^bb1
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}
Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  br ^bb1
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  br ^bb1
^bb1:  // pred: ^bb0
  ts.ReturnInternal
}


Trying to match ""
"" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>"
"`anonymous-namespace'::RemoveUnused<class mlir::typescript::LoadOp>" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


Trying to match "`anonymous-namespace'::NormalizeCast"
"`anonymous-namespace'::NormalizeCast" result 1
// *** IR Dump After Pattern Application ***
ts.Func @main () -> ()  {
  %0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %1 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %3 = ts.Constant {value = "number"} : !ts.string
  %4 = ts.CreateUnionInstance %2, %3 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %0 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%0) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %1 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


The pattern rewrite doesn't converge after scanning 10 times

Insert const at: 
%0 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>

const to insert: 
%2 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
** Insert  : 'ts.Constant'(0x1b285ccc9e0)

const to insert: 
%3 = ts.Constant {value = "number"} : !ts.string
** Insert  : 'ts.Constant'(0x1b285cca320)

!! AFTER CONST RELOC FUNC DUMP: 
ts.Func @main () -> ()  {
  %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %1 = ts.Constant {value = "number"} : !ts.string
  %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}
Outlined 0 functions built from async.execute operations

//===-------------------------------------------===//
Legalizing operation : 'module'(0x1b285d063e0) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x1b285d068b0) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x1b285ccc9e0) {
  %0 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x1b285cca320) {
  %1 = "ts.Constant"() {value = "number"} : () -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cc9d80) {
  %2 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cca780) {
  %3 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x1b285ca2600) {
  %4 = "ts.CreateUnionInstance"(%0, %1) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c67b80) {
  "ts.Store"(%4, %2) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x1b285c13f90) {
  %5 = "ts.Load"(%2) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x1b285c117d0) {
  %6 = "ts.Cast"(%5) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.boolean,!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c679e0) {
  "ts.Store"(%6, %3) : (!ts.union<!ts.boolean,!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.ReturnInternal'(0x1b285d402c0) {
  "ts.ReturnInternal"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'module'(0x1b285d063e0) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x1b285d068b0) {
  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x1b285ccc9e0) {
  %0 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x1b285cca320) {
  %1 = "ts.Constant"() {value = "number"} : () -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cc9d80) {
  %2 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cca780) {
  %3 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x1b285ca2600) {
  %4 = "ts.CreateUnionInstance"(%0, %1) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c67b80) {
  "ts.Store"(%4, %2) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x1b285c13f90) {
  %5 = "ts.Load"(%2) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x1b285c117d0) {
  %6 = "ts.Cast"(%5) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.boolean,!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c679e0) {
  "ts.Store"(%6, %3) : (!ts.union<!ts.boolean,!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.ReturnInternal'(0x1b285d402c0) {
  "ts.ReturnInternal"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold
} -> FAILURE : no matched legalization pattern
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d05c50) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d07040) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d070f0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d070f0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d045a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d06330) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d06330) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06b70) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d06b70) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05f10) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d04c80) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d04c80) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05150) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d05200) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d05200) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d052b0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d065f0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d065f0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05360) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d05360) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05570) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d05620) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d05620) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06c20) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d066a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d04650) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d04650) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d058e0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d058e0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06750) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d06960) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FunctionLikeSignatureConversion"
"`anonymous-namespace'::FunctionLikeSignatureConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d06960) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'llvm.func'(0x1b285d05990) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d05ba0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

!! BEFORE DUMP: 
module @"c:\\temp\\1.ts"  {
  ts.Func @main () -> ()  {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'module'(0x1b285d063e0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Func'(0x1b285d068b0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpLowering"
    ** Insert  : 'func'(0x1b285d0e890)
    ** Erase   : 'ts.Func'(0x1b285d068b0)
"`anonymous-namespace'::FuncOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'func'(0x1b285d0e890) {
      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
        ** Insert  : 'llvm.func'(0x1b285d0eaa0)
        ** Erase   : 'func'(0x1b285d0e890)
"`anonymous-namespace'::FuncOpConversion" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.func'(0x1b285d0eaa0) {
        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @main() {
    %0 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
    %1 = ts.Constant {value = "number"} : !ts.string
    %2 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %3 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
    %4 = ts.CreateUnionInstance %0, %1 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
    ts.Store %4, %2 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
    %5 = ts.Load(%2) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
    %6 = ts.Cast %5 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    ts.Store %6, %3 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'ts.Constant'(0x1b285ccc9e0) {
  %0 = "ts.Constant"() {value = 1.000000e+01 : f64} : () -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Constant -> ()' {
Trying to match "`anonymous-namespace'::ConstantOpLowering"
    ** Insert  : 'std.constant'(0x1b285ccdac0)
    ** Replace : 'ts.Constant'(0x1b285ccc9e0)
"`anonymous-namespace'::ConstantOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'std.constant'(0x1b285ccdac0) {
      %0 = "std.constant"() {value = 1.000000e+01 : f64} : () -> f64

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'std.constant -> ()' {
Trying to match "`anonymous-namespace'::ConstantOpLowering"
        ** Insert  : 'llvm.mlir.constant'(0x1b285cce060)
        ** Replace : 'std.constant'(0x1b285ccdac0)
"`anonymous-namespace'::ConstantOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x1b285cce060) {
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
  %4 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %5 = ts.CreateUnionInstance %1, %2 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %3 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%3) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %7, %4 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %4 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %5 = ts.CreateUnionInstance %1, %2 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %5, %3 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %6 = ts.Load(%3) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %7 = ts.Cast %6 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %7, %4 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Constant'(0x1b285cca320) {
  %3 = "ts.Constant"() {value = "number"} : () -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Constant -> ()' {
Trying to match "`anonymous-namespace'::ConstantOpLowering"
    ** Insert  : 'llvm.mlir.global'(0x1b285d0f230)
    ** Insert  : 'llvm.mlir.addressof'(0x1b285ccf3c0)
    ** Insert  : 'llvm.mlir.constant'(0x1b285ccdc00)
    ** Insert  : 'llvm.getelementptr'(0x1b285b74830)
    ** Replace : 'ts.Constant'(0x1b285cca320)
"`anonymous-namespace'::ConstantOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.global'(0x1b285d0f230) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.addressof'(0x1b285ccf3c0) {
      %3 = "llvm.mlir.addressof"() {global_name = @s_9237349086447201248} : () -> !llvm.ptr<array<7 x i8>>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.constant'(0x1b285ccdc00) {
      %4 = "llvm.mlir.constant"() {value = 0 : i64} : () -> i64

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.getelementptr'(0x1b285b74830) {
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
  %7 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %8 = ts.CreateUnionInstance %1, %5 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %8, %6 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %9 = ts.Load(%6) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %10 = ts.Cast %9 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %10, %7 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cc9d80) {
  %7 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.number,!ts.string> is captured: 0
    ** Insert  : 'llvm.mlir.constant'(0x1b285ccf5a0)
    ** Insert  : 'llvm.alloca'(0x1b285c11410)
    ** Replace : 'ts.Variable'(0x1b285cc9d80)
"`anonymous-namespace'::VariableOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.constant'(0x1b285ccf5a0) {
      %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.alloca'(0x1b285c11410) {
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
  %9 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %10 = ts.CreateUnionInstance %3, %7 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %10, %8 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %11 = ts.Load(%8) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %12 = ts.Cast %11 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %12, %9 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Variable'(0x1b285cca780) {
  %10 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.boolean,!ts.number,!ts.string> is captured: 0
    ** Insert  : 'llvm.mlir.constant'(0x1b285ccef60)
    ** Insert  : 'llvm.alloca'(0x1b285c135d0)
    ** Replace : 'ts.Variable'(0x1b285cca780)
"`anonymous-namespace'::VariableOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.constant'(0x1b285ccef60) {
      %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.alloca'(0x1b285c135d0) {
      %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %11 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %12 = ts.CreateUnionInstance %5, %9 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  ts.Store %12, %10 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %13 = ts.Load(%10) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %14 = ts.Cast %13 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %14, %11 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.CreateUnionInstance'(0x1b285ca2600) {
  %13 = "ts.CreateUnionInstance"(%6, %10) : (!ts.number, !ts.string) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.CreateUnionInstance -> ()' {
Trying to match "`anonymous-namespace'::CreateUnionInstanceOpLowering"
    ** Insert  : 'llvm.mlir.undef'(0x1b285ccd340)
    ** Insert  : 'llvm.insertvalue'(0x1b285ca7300)
    ** Insert  : 'llvm.insertvalue'(0x1b285ca7bc0)
    ** Insert  : 'ts.Variable'(0x1b285c12a90)
    ** Insert  : 'ts.Variable'(0x1b285ccd160)
    ** Insert  : 'ts.MemoryCopy'(0x1b285c68200)
    ** Insert  : 'ts.Load'(0x1b285c111d0)
    ** Replace : 'ts.CreateUnionInstance'(0x1b285ca2600)
"`anonymous-namespace'::CreateUnionInstanceOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.mlir.undef'(0x1b285ccd340) {
      %13 = "llvm.mlir.undef"() : () -> !llvm.struct<(ptr<i8>, f64)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.insertvalue'(0x1b285ca7300) {
      %14 = "llvm.insertvalue"(%13, %10) {position = [0 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, !ts.string) -> !llvm.struct<(ptr<i8>, f64)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.insertvalue'(0x1b285ca7bc0) {
      %15 = "llvm.insertvalue"(%14, %6) {position = [1 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, !ts.number) -> !llvm.struct<(ptr<i8>, f64)>

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Variable'(0x1b285c12a90) {
      %16 = "ts.Variable"(%15) {captured = false} : (!llvm.struct<(ptr<i8>, f64)>) -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !llvm.struct<(ptr<i8>, f64)> is captured: 0
        ** Insert  : 'llvm.mlir.constant'(0x1b285ccd3e0)
        ** Insert  : 'llvm.alloca'(0x1b285c13150)
        ** Insert  : 'llvm.store'(0x1b285c67840)
        ** Replace : 'ts.Variable'(0x1b285c12a90)
"`anonymous-namespace'::VariableOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x1b285ccd3e0) {
          %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.alloca'(0x1b285c13150) {
          %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.store'(0x1b285c67840) {
          "llvm.store"(%17, %1) : (!llvm.struct<(ptr<i8>, f64)>, !llvm.ptr<struct<(ptr<i8>, f64)>>) -> ()

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//
      } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %13 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %22 = ts.Cast %21 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %22, %13 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Variable'(0x1b285ccd160) {
      %19 = "ts.Variable"() {captured = false} : () -> !ts.ref<!ts.union<!ts.number,!ts.string>>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Variable -> ()' {
Trying to match "`anonymous-namespace'::VariableOpLowering"

!! variable allocation: !ts.union<!ts.number,!ts.string> is captured: 0
        ** Insert  : 'llvm.mlir.constant'(0x1b285ccdd40)
        ** Insert  : 'llvm.alloca'(0x1b285c11710)
        ** Replace : 'ts.Variable'(0x1b285ccd160)
"`anonymous-namespace'::VariableOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x1b285ccdd40) {
          %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.alloca'(0x1b285c11710) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %24 = ts.Cast %23 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %24, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.MemoryCopy'(0x1b285c68200) {
      "ts.MemoryCopy"(%21, %20) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.MemoryCopy -> ()' {
Trying to match "`anonymous-namespace'::MemoryCopyOpLowering"
        ** Insert  : 'llvm.func'(0x1b285d0cbb0)
        ** Insert  : 'llvm.bitcast'(0x1b285c12fd0)
        ** Insert  : 'llvm.bitcast'(0x1b285c11890)
        ** Insert  : 'ts.SizeOf'(0x1b285ccda20)
        ** Insert  : 'ts.SizeOf'(0x1b285ccf500)
        ** Insert  : 'llvm.icmp'(0x1b285ca67a0)
        ** Insert  : 'llvm.select'(0x1b285b72730)
        ** Insert  : 'llvm.mlir.constant'(0x1b285ccd700)
        ** Insert  : 'llvm.call'(0x1b285ba5830)
        ** Erase   : 'ts.MemoryCopy'(0x1b285c68200)
"`anonymous-namespace'::MemoryCopyOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.func'(0x1b285d0cbb0) {
        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.bitcast'(0x1b285c12fd0) {
          %22 = "llvm.bitcast"(%1) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<i8>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.bitcast'(0x1b285c11890) {
          %23 = "llvm.bitcast"(%3) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> !llvm.ptr<i8>

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'ts.SizeOf'(0x1b285ccda20) {
          %24 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64

          * Fold {
          } -> FAILURE : unable to fold

          * Pattern : 'ts.SizeOf -> ()' {
Trying to match "`anonymous-namespace'::SizeOfOpLowering"
            ** Insert  : 'llvm.mlir.null'(0x1b285cce380)

!! size of - storage type: [!llvm.struct<(ptr<i8>, f64)>] llvm storage type: [!llvm.struct<(ptr<i8>, f64)>] llvm ptr: [!llvm.ptr<struct<(ptr<i8>, f64)>>] value: [%23 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>]
            ** Insert  : 'llvm.mlir.constant'(0x1b285ccd7a0)
            ** Insert  : 'llvm.getelementptr'(0x1b285ca8f00)
            ** Insert  : 'llvm.ptrtoint'(0x1b285c11950)
            ** Replace : 'ts.SizeOf'(0x1b285ccda20)
"`anonymous-namespace'::SizeOfOpLowering" result 1

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.null'(0x1b285cce380) {
              %24 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, f64)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.constant'(0x1b285ccd7a0) {
              %25 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.getelementptr'(0x1b285ca8f00) {
              %26 = "llvm.getelementptr"(%24, %25) : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.ptrtoint'(0x1b285c11950) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %35 = ts.Cast %34 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %35, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


        } -> SUCCESS
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'ts.SizeOf'(0x1b285ccf500) {
          %29 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64

          * Fold {
          } -> FAILURE : unable to fold

          * Pattern : 'ts.SizeOf -> ()' {
Trying to match "`anonymous-namespace'::SizeOfOpLowering"
            ** Insert  : 'llvm.mlir.null'(0x1b285ccdde0)

!! size of - storage type: [!llvm.struct<(ptr<i8>, ptr<i8>)>] llvm storage type: [!llvm.struct<(ptr<i8>, ptr<i8>)>] llvm ptr: [!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>] value: [%28 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>]
            ** Insert  : 'llvm.mlir.constant'(0x1b285ccea60)
            ** Insert  : 'llvm.getelementptr'(0x1b285ca5ee0)
            ** Insert  : 'llvm.ptrtoint'(0x1b285c11b90)
            ** Replace : 'ts.SizeOf'(0x1b285ccf500)
"`anonymous-namespace'::SizeOfOpLowering" result 1

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.null'(0x1b285ccdde0) {
              %29 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.mlir.constant'(0x1b285ccea60) {
              %30 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.getelementptr'(0x1b285ca5ee0) {
              %31 = "llvm.getelementptr"(%29, %30) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

            } -> SUCCESS : operation marked legal by the target
            //===-------------------------------------------===//

            //===-------------------------------------------===//
            Legalizing operation : 'llvm.ptrtoint'(0x1b285c11b90) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %39 = ts.Cast %38 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %39, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


        } -> SUCCESS
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.icmp'(0x1b285ca67a0) {
          %34 = "llvm.icmp"(%28, %33) {predicate = 6 : i64} : (i64, i64) -> i1

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.select'(0x1b285b72730) {
          %35 = "llvm.select"(%34, %28, %33) : (i1, i64, i64) -> i64

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.mlir.constant'(0x1b285ccd700) {
          %36 = "llvm.mlir.constant"() {value = false} : () -> i1

        } -> SUCCESS : operation marked legal by the target
        //===-------------------------------------------===//

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.call'(0x1b285ba5830) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %39 = ts.Cast %38 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %39, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


    } -> SUCCESS
    //===-------------------------------------------===//

    //===-------------------------------------------===//
    Legalizing operation : 'ts.Load'(0x1b285c111d0) {
      %37 = "ts.Load"(%21) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

      * Fold {
      } -> FAILURE : unable to fold

      * Pattern : 'ts.Load -> ()' {
Trying to match "`anonymous-namespace'::LoadOpLowering"
        ** Insert  : 'llvm.load'(0x1b285c12d90)
        ** Replace : 'ts.Load'(0x1b285c111d0)
"`anonymous-namespace'::LoadOpLowering" result 1

        //===-------------------------------------------===//
        Legalizing operation : 'llvm.load'(0x1b285c12d90) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %40 = ts.Cast %39 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %40, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %40 = ts.Cast %39 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %40, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c67b80) {
  "ts.Store"(%39, %15) : (!ts.union<!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Store -> ()' {
Trying to match "`anonymous-namespace'::StoreOpLowering"
    ** Insert  : 'llvm.store'(0x1b285c670f0)
    ** Replace : 'ts.Store'(0x1b285c67b80)
"`anonymous-namespace'::StoreOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.store'(0x1b285c670f0) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %40 = ts.Cast %39 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %40, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Load'(0x1b285c13f90) {
  %40 = "ts.Load"(%15) : (!ts.ref<!ts.union<!ts.number,!ts.string>>) -> !ts.union<!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Load -> ()' {
Trying to match "`anonymous-namespace'::LoadOpLowering"
    ** Insert  : 'llvm.load'(0x1b285c12550)
    ** Replace : 'ts.Load'(0x1b285c13f90)
"`anonymous-namespace'::LoadOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.load'(0x1b285c12550) {
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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Cast'(0x1b285c117d0) {
  %42 = "ts.Cast"(%41) : (!ts.union<!ts.number,!ts.string>) -> !ts.union<!ts.boolean,!ts.number,!ts.string>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Cast -> ()' {
Trying to match "`anonymous-namespace'::CastOpLowering"
    ** Replace : 'ts.Cast'(0x1b285c117d0)
"`anonymous-namespace'::CastOpLowering" result 1
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.Store'(0x1b285c679e0) {
  "ts.Store"(%42, %16) : (!ts.union<!ts.boolean,!ts.number,!ts.string>, !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>) -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.Store -> ()' {
Trying to match "`anonymous-namespace'::StoreOpLowering"
    ** Insert  : 'llvm.store'(0x1b285c66730)
    ** Replace : 'ts.Store'(0x1b285c679e0)
"`anonymous-namespace'::StoreOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.store'(0x1b285c66730) {
      "llvm.store"(%40, %5) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'ts.ReturnInternal'(0x1b285d402c0) {
  "ts.ReturnInternal"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.ReturnInternal -> ()' {
Trying to match "`anonymous-namespace'::ReturnInternalOpLowering"
    ** Insert  : 'llvm.return'(0x1b285d3e730)
    ** Replace : 'ts.ReturnInternal'(0x1b285d402c0)
"`anonymous-namespace'::ReturnInternalOpLowering" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.return'(0x1b285d3e730) {
      "llvm.return"() : () -> ()

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
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
  %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  llvm.return
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'func'(0x1b285d05c50) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0ef70)
    ** Erase   : 'func'(0x1b285d05c50)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0ef70) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d07040) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0d810)
    ** Erase   : 'func'(0x1b285d07040)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0d810) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d070f0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e1b0)
    ** Erase   : 'func'(0x1b285d070f0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e1b0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d045a0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0ecb0)
    ** Erase   : 'func'(0x1b285d045a0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0ecb0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06330) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e730)
    ** Erase   : 'func'(0x1b285d06330)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e730) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06b70) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e050)
    ** Erase   : 'func'(0x1b285d06b70)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e050) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05f10) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0f0d0)
    ** Erase   : 'func'(0x1b285d05f10)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0f0d0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d04c80) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0f180)
    ** Erase   : 'func'(0x1b285d04c80)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0f180) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05150) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0dd90)
    ** Erase   : 'func'(0x1b285d05150)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0dd90) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05200) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0f440)
    ** Erase   : 'func'(0x1b285d05200)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0f440) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d052b0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e520)
    ** Erase   : 'func'(0x1b285d052b0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e520) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d065f0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e260)
    ** Erase   : 'func'(0x1b285d065f0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e260) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05360) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0eec0)
    ** Erase   : 'func'(0x1b285d05360)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0eec0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05570) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0cf20)
    ** Erase   : 'func'(0x1b285d05570)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0cf20) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d05620) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0f4f0)
    ** Erase   : 'func'(0x1b285d05620)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0f4f0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06c20) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e7e0)
    ** Erase   : 'func'(0x1b285d06c20)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e7e0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d066a0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0c9a0)
    ** Erase   : 'func'(0x1b285d066a0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0c9a0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d04650) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0d8c0)
    ** Erase   : 'func'(0x1b285d04650)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0d8c0) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d058e0) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0ed60)
    ** Erase   : 'func'(0x1b285d058e0)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0ed60) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06750) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0e310)
    ** Erase   : 'func'(0x1b285d06750)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0e310) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'func'(0x1b285d06960) {
  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func -> ()' {
Trying to match "`anonymous-namespace'::FuncOpConversion"
    ** Insert  : 'llvm.func'(0x1b285d0cc60)
    ** Erase   : 'func'(0x1b285d06960)
"`anonymous-namespace'::FuncOpConversion" result 1

    //===-------------------------------------------===//
    Legalizing operation : 'llvm.func'(0x1b285d0cc60) {
    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
    %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
    %41 = ts.Cast %40 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
    llvm.store %39, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    ts.Store %41, %15 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
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
Legalizing operation : 'llvm.func'(0x1b285d05990) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d05ba0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

!! SourceMaterialization: loc:[ loc("c:\\temp\\1.ts":5:9) ] result: [ !ts.number ]

!! SourceMaterialization value: [ %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64 ]
** Insert  : 'ts.DialectCast'(0x1b285c13c90)

//===-------------------------------------------===//
Legalizing operation : 'ts.DialectCast'(0x1b285c13c90) {
  %10 = "ts.DialectCast"(%8) : (f64) -> !ts.number

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.DialectCast -> ()' {
Trying to match "`anonymous-namespace'::DialectCastOpLowering"
    ** Replace : 'ts.DialectCast'(0x1b285c13c90)
"`anonymous-namespace'::DialectCastOpLowering" result 1
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.DialectCast %8 : f64 to !ts.number
  %10 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %11 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %12 = llvm.mlir.constant(0 : i64) : i64
  %13 = llvm.getelementptr %11[%12, %12] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %14 = ts.Constant {value = "number"} : !ts.string
  %15 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %16 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %17 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %18 = llvm.insertvalue %14, %17[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %19 = llvm.insertvalue %10, %18[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %19, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %20 = ts.Variable(%19) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %21 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %22 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %23 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
  %24 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %25 = llvm.mlir.constant(1 : i64) : i64
  %26 = llvm.getelementptr %24[%25] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %27 = llvm.ptrtoint %26 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
  %28 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, f64)>} : () -> i64
  %29 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %30 = llvm.mlir.constant(1 : i64) : i64
  %31 = llvm.getelementptr %29[%30] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %32 = llvm.ptrtoint %31 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
  %33 = "ts.SizeOf"() {type = !llvm.struct<(ptr<i8>, ptr<i8>)>} : () -> i64
  %34 = llvm.icmp "ult" %28, %33 : i64
  %35 = llvm.select %34, %28, %33 : i1, i64
  %36 = llvm.mlir.constant(false) : i1
  llvm.call @llvm.memcpy.p0i8.p0i8.i64(%22, %23, %35, %36) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  "ts.MemoryCopy"(%21, %20) : (!ts.ref<!ts.union<!ts.number,!ts.string>>, !ts.ref<!llvm.struct<(ptr<i8>, f64)>>) -> ()
  %37 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %38 = ts.Load(%21) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %39 = ts.CreateUnionInstance %10, %14 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %37, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %39, %15 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %40 = llvm.load %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %41 = ts.Load(%15) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %42 = ts.Cast %41 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  llvm.store %40, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %42, %16 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  llvm.return
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

!! SourceMaterialization: loc:[ loc("c:\\temp\\1.ts":5:5) ] result: [ !ts.string ]

!! SourceMaterialization value: [ %13 = llvm.getelementptr %11[%12, %12] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8> ]
** Insert  : 'ts.DialectCast'(0x1b285c11650)

//===-------------------------------------------===//
Legalizing operation : 'ts.DialectCast'(0x1b285c11650) {
  %15 = "ts.DialectCast"(%14) : (!llvm.ptr<i8>) -> !ts.string

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'ts.DialectCast -> ()' {
Trying to match "`anonymous-namespace'::DialectCastOpLowering"
    ** Replace : 'ts.DialectCast'(0x1b285c11650)
"`anonymous-namespace'::DialectCastOpLowering" result 1
  } -> SUCCESS : pattern applied successfully
// *** IR Dump After Pattern Application ***
llvm.func @main() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %6 = llvm.mlir.constant(1 : i32) : i32
  %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
  %cst = constant 1.000000e+01 : f64
  %9 = ts.DialectCast %8 : f64 to !ts.number
  %10 = ts.Constant {value = 1.000000e+01 : f64} : !ts.number
  %11 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
  %12 = llvm.mlir.constant(0 : i64) : i64
  %13 = llvm.getelementptr %11[%12, %12] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
  %14 = ts.DialectCast %13 : !llvm.ptr<i8> to !ts.string
  %15 = ts.Constant {value = "number"} : !ts.string
  %16 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %17 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  %18 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
  %19 = llvm.insertvalue %15, %18[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
  %20 = llvm.insertvalue %10, %19[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
  llvm.store %20, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
  %21 = ts.Variable(%20) {false} : !llvm.struct<(ptr<i8>, f64)> -> !ts.ref<!llvm.struct<(ptr<i8>, f64)>>
  %22 = ts.Variable() {false} :  -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %23 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
  %24 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
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
  %38 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %39 = ts.Load(%22) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %40 = ts.CreateUnionInstance %10, %15 : !ts.number, !ts.string to !ts.union<!ts.number,!ts.string>
  llvm.store %38, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %40, %16 : !ts.union<!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.number,!ts.string>>
  %41 = llvm.load %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  %42 = ts.Load(%16) : !ts.ref<!ts.union<!ts.number,!ts.string>> -> !ts.union<!ts.number,!ts.string>
  %43 = ts.Cast %42 : !ts.union<!ts.number,!ts.string> to !ts.union<!ts.boolean,!ts.number,!ts.string>
  llvm.store %41, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
  ts.Store %43, %17 : !ts.union<!ts.boolean,!ts.number,!ts.string> -> !ts.ref<!ts.union<!ts.boolean,!ts.number,!ts.string>>
  llvm.return
  ts.ReturnInternal
}


} -> SUCCESS
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'module'(0x1b285d063e0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0cbb0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.global'(0x1b285d0f230) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0eaa0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccdd40) {
  %0 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x1b285c11710) {
  %1 = "llvm.alloca"(%0) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccd3e0) {
  %2 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x1b285c13150) {
  %3 = "llvm.alloca"(%2) : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccef60) {
  %4 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x1b285c135d0) {
  %5 = "llvm.alloca"(%4) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccf5a0) {
  %6 = "llvm.mlir.constant"() {value = 1 : i32} : () -> i32

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.alloca'(0x1b285c11410) {
  %7 = "llvm.alloca"(%6) : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285cce060) {
  %8 = "llvm.mlir.constant"() {value = 1.000000e+01 : f64} : () -> f64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.addressof'(0x1b285ccf3c0) {
  %9 = "llvm.mlir.addressof"() {global_name = @s_9237349086447201248} : () -> !llvm.ptr<array<7 x i8>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccdc00) {
  %10 = "llvm.mlir.constant"() {value = 0 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x1b285b74830) {
  %11 = "llvm.getelementptr"(%9, %10, %10) : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.undef'(0x1b285ccd340) {
  %12 = "llvm.mlir.undef"() : () -> !llvm.struct<(ptr<i8>, f64)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.insertvalue'(0x1b285ca7300) {
  %13 = "llvm.insertvalue"(%12, %11) {position = [0 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, !llvm.ptr<i8>) -> !llvm.struct<(ptr<i8>, f64)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.insertvalue'(0x1b285ca7bc0) {
  %14 = "llvm.insertvalue"(%13, %8) {position = [1 : i32]} : (!llvm.struct<(ptr<i8>, f64)>, f64) -> !llvm.struct<(ptr<i8>, f64)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x1b285c67840) {
  "llvm.store"(%14, %3) : (!llvm.struct<(ptr<i8>, f64)>, !llvm.ptr<struct<(ptr<i8>, f64)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.bitcast'(0x1b285c12fd0) {
  %15 = "llvm.bitcast"(%1) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.bitcast'(0x1b285c11890) {
  %16 = "llvm.bitcast"(%3) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> !llvm.ptr<i8>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.null'(0x1b285cce380) {
  %17 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, f64)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccd7a0) {
  %18 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x1b285ca8f00) {
  %19 = "llvm.getelementptr"(%17, %18) : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.ptrtoint'(0x1b285c11950) {
  %20 = "llvm.ptrtoint"(%19) : (!llvm.ptr<struct<(ptr<i8>, f64)>>) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.null'(0x1b285ccdde0) {
  %21 = "llvm.mlir.null"() : () -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccea60) {
  %22 = "llvm.mlir.constant"() {value = 1 : i64} : () -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.getelementptr'(0x1b285ca5ee0) {
  %23 = "llvm.getelementptr"(%21, %22) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.ptrtoint'(0x1b285c11b90) {
  %24 = "llvm.ptrtoint"(%23) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.icmp'(0x1b285ca67a0) {
  %25 = "llvm.icmp"(%20, %24) {predicate = 6 : i64} : (i64, i64) -> i1

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.select'(0x1b285b72730) {
  %26 = "llvm.select"(%25, %20, %24) : (i1, i64, i64) -> i64

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.mlir.constant'(0x1b285ccd700) {
  %27 = "llvm.mlir.constant"() {value = false} : () -> i1

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.call'(0x1b285ba5830) {
  "llvm.call"(%15, %16, %26, %27) {callee = @llvm.memcpy.p0i8.p0i8.i64} : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.load'(0x1b285c12d90) {
  %28 = "llvm.load"(%1) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.struct<(ptr<i8>, ptr<i8>)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x1b285c670f0) {
  "llvm.store"(%28, %7) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.load'(0x1b285c12550) {
  %29 = "llvm.load"(%7) : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> !llvm.struct<(ptr<i8>, ptr<i8>)>

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.store'(0x1b285c66730) {
  "llvm.store"(%29, %5) : (!llvm.struct<(ptr<i8>, ptr<i8>)>, !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>) -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.return'(0x1b285d3e730) {
  "llvm.return"() : () -> ()

} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0ef70) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0d810) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e1b0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0ecb0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e730) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e050) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0f0d0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0f180) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0dd90) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0f440) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e520) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e260) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0eec0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0cf20) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0f4f0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e7e0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0c9a0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0d8c0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0ed60) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0e310) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d0cc60) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d05990) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

//===-------------------------------------------===//
Legalizing operation : 'llvm.func'(0x1b285d05ba0) {
} -> SUCCESS : operation marked legal by the target
//===-------------------------------------------===//

!! AFTER DUMP: 
module @"c:\\temp\\1.ts"  {
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %9 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.getelementptr %9[%10, %10] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %13 = llvm.insertvalue %11, %12[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %14 = llvm.insertvalue %8, %13[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %14, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %15 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %16 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %17 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %20 = llvm.ptrtoint %19 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %21 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %25 = llvm.icmp "ult" %20, %24 : i64
    %26 = llvm.select %25, %20, %24 : i1, i64
    %27 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%15, %16, %26, %27) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %28 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %28, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %29 = llvm.load %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %29, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
** Insert  : 'llvm.func'(0x1b285d0d600)
** Insert  : 'llvm.call'(0x1b285d3efa0)
module @"c:\\temp\\1.ts"  {
  llvm.func @GC_init()
  llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  llvm.mlir.global internal constant @s_9237349086447201248("number\00")
  llvm.func @main() {
    llvm.call @GC_init() : () -> ()
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %2 x !llvm.struct<(ptr<i8>, f64)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.alloca %4 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.alloca %6 x !llvm.struct<(ptr<i8>, ptr<i8>)> : (i32) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %8 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %9 = llvm.mlir.addressof @s_9237349086447201248 : !llvm.ptr<array<7 x i8>>
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.getelementptr %9[%10, %10] : (!llvm.ptr<array<7 x i8>>, i64, i64) -> !llvm.ptr<i8>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<i8>, f64)>
    %13 = llvm.insertvalue %11, %12[0 : i32] : !llvm.struct<(ptr<i8>, f64)>
    %14 = llvm.insertvalue %8, %13[1 : i32] : !llvm.struct<(ptr<i8>, f64)>
    llvm.store %14, %3 : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %15 = llvm.bitcast %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to !llvm.ptr<i8>
    %16 = llvm.bitcast %3 : !llvm.ptr<struct<(ptr<i8>, f64)>> to !llvm.ptr<i8>
    %17 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, f64)>>
    %18 = llvm.mlir.constant(1 : i64) : i64
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<struct<(ptr<i8>, f64)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, f64)>>
    %20 = llvm.ptrtoint %19 : !llvm.ptr<struct<(ptr<i8>, f64)>> to i64
    %21 = llvm.mlir.null : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.getelementptr %21[%22] : (!llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>, i64) -> !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %24 = llvm.ptrtoint %23 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>> to i64
    %25 = llvm.icmp "ult" %20, %24 : i64
    %26 = llvm.select %25, %20, %24 : i1, i64
    %27 = llvm.mlir.constant(false) : i1
    llvm.call @llvm.memcpy.p0i8.p0i8.i64(%15, %16, %26, %27) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    %28 = llvm.load %1 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %28, %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    %29 = llvm.load %7 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
    llvm.store %29, %5 : !llvm.ptr<struct<(ptr<i8>, ptr<i8>)>>
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
