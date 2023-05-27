
class A {
    propA: number;
}

class B {
    propB: number;
}

class C extends A {
    propC: number;
}

declare function isA(p1: any): p1 is A;
declare function isB(p1: any): p1 is B;
declare function isC(p1: any): p1 is C;

declare function retC(): C;

// Call signature
interface I1 {
    (p1: A): p1 is C;
}

// The parameter index and argument index for the type guard target is matching.
// The type predicate type is assignable to the parameter type.
declare function isC_multipleParams(p1, p2): p1 is C;

class D {
    method1(p1: A): p1 is C {
        return true;
    }
}

// Function type
declare function f2(p1: (p1: A) => p1 is C);

// Evaluations are asssignable to boolean.
declare function acceptingBoolean(a: boolean);

// Type predicates with different parameter name.
declare function acceptingTypeGuardFunction(p1: (item) => item is A);

function main() {

    let a: A = new A();
    let b: B = new B();

    // Basic
    if (isC(a)) {
        a.propC;
    }

    // Sub type
    let subType: C = new C();
    if (isA(subType)) {
        subType.propA;
    }

    // Union type
    let union: A | B = new A();
    if (isA(union)) {
        union.propA;
    }

    if (isC_multipleParams(a, 0)) {
        a.propC;
    }

    // Methods
    let obj: {
        func1(p1: A): p1 is C;
    }

    // Arrow function
    let f1 = (p1: A): p1 is C => false;

    // Function expressions
    f2(function (p1: A): p1 is C {
        return true;
    });

    acceptingBoolean(isA(a));

    acceptingTypeGuardFunction(isA);

    // Binary expressions
    let union2: C | B = new B();
    let union3: boolean | B = isA(union2) || union2;

    print("done.");
}