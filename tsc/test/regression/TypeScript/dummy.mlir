// RUN: tsc-opt %s | tsc-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: ts.Print %{{.*}} : i32
        ts.Print %0 : i32
        return
    }
}
