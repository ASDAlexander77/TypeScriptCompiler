// A `declare var` is a global defined elsewhere, also when the module has code of its own, which runs in a
// function of its own: functions read @cc, which was never declared ("'llvm.mlir.addressof' op must reference
// a global"). Compiled to an object only: nothing defines `cc`.
declare var cc: { log(msg?: string): void; };
class Foo { constructor() { cc.log("a"); } }
new Foo();
