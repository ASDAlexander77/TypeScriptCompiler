class A {
    propA: number;
}

class B {
    propB: number;
}

class C extends A {
    propC: number;
}

// TODO: union is not working here, why?
function isA(p1: any): p1 is A {
    return p1 instanceof A;
}

function isB(p1: A | B | C): p1 is B {
    return p1 instanceof B;
}

function isC(p1: A | B | C): p1 is C {
    return p1 instanceof C;
}

function main() {

    //let abc: A | B | C = new A();
    let abc = new A();

    if (isA(abc)) {
        print("this is A");
        abc.propA;
    }

    if (isB(abc)) {
        print("this is B");
        abc.propB;
    }

    if (isC(abc)) {
        print("this is C");
        abc.propC;
    }

    print("done.");
}