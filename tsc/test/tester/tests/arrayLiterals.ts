class C { private p; }

// Contextual type C with numeric index signature of type Base makes array literal of Derived have type Base[]
class Base { private p; }
class Derived1 extends Base { private m };
class Derived2 extends Base { private n };

function main() {
    // Empty array literal with no contextual type has type Undefined[]
    let arr1 = [[], [1], ['']];

    let arr2 = [[null], [1], ['']];


    // Array literal with elements of only EveryType E has type E[]
    let stringArrArr = [[''], [""]];

    let stringArr = ['', ""];

    let numberArr = [0, 0.0, 0x00, 1e1];

    let boolArr = [false, true, false, true];


    let classArr = [new C(), new C()];

    let classTypeArray2: Array<typeof C>; // Should OK, not be a parse error

    // Contextual type C with numeric index signature makes array literal of EveryType E of type BCT(E,C)[]
    //let context1: { [n: number]: { a: string; b: number; }; } = [{ a: '', b: 0, c: '' }, { a: "", b: 3, c: 0 }];
    let context2 = [{ a: '', b: 0, c: '' }, { a: "", b: 3, c: 0 }];


    let context3: Base[] = [new Derived1(), new Derived2()];

    // Contextual type C with numeric index signature of type Base makes array literal of Derived1 and Derived2 have type Base[]
    let context4: Base[] = [new Derived1(), new Derived1()];

    print("done.");
}