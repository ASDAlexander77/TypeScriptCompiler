// Reading the value an untyped `catch (e)` or a `catch (e: any)` binds.
//
// Such a catch binds an `any`, but a thrown number, string or class instance is not one, and the
// catch variable used to receive the raw value: the first read of it as an `any` crashed, even
// for `throw "message"`. Now the value is boxed on its way in:
//  - Windows: the thrown value's `.PEAX` CatchableType names a copy function that boxes it, and
//    the CRT calls that instead of copying bytes (copyThunkPrefix, LLVMRTTIHelperVCWin32Const.h);
//  - Linux: the catch is a real catch-all, and binding it compares the exception's type_info with
//    each type tslang throws (linux::SaveCatchVarOpLowering). The `_ZTIPv` clause it used was not
//    a catch-all at all, so a thrown number was never caught.
//
// Also here: `catch (e: number)` takes `throw 1`. An integer literal is an int, so the int's
// throw lists a converting entry for `.N` on Windows, and the catch lists the int type_info on
// Linux.

class Failure {
    code = 42;
}

class Derived extends Failure {
    extra = 1;
}

function untypedNumber() {
    let r: number = 0;
    try {
        throw 2.5;
    } catch (e) {
        r = <number>e;
    }

    return r;
}

function untypedInt() {
    let r = 0;
    try {
        throw 7;
    } catch (e) {
        r = <int>e;
    }

    return r;
}

function untypedString() {
    let r = "";
    try {
        throw "boom";
    } catch (e) {
        r = <string>e;
    }

    return r;
}

function untypedClass() {
    let r = 0;
    try {
        throw new Failure();
    } catch (e) {
        r = (<Failure>e).code;
    }

    return r;
}

function untypedDerivedAsBase() {
    let r = 0;
    try {
        throw new Derived();
    } catch (e) {
        r = (<Failure>e).code + (<Derived>e).extra;
    }

    return r;
}

function anyTyped() {
    let r = 0;
    try {
        throw 11;
    } catch (e: any) {
        r = <int>e;
    }

    return r;
}

function alreadyAny() {
    let r = 0;
    try {
        throw <any>123;
    } catch (e) {
        r = <int>e;
    }

    return r;
}

function numberCatchesInt() {
    let r: number = 0;
    try {
        throw 5;
    } catch (e: number) {
        r = e + 0.5;
    }

    return r;
}

function typedCatchesUnchanged() {
    let sum: number = 0;
    try {
        throw 3;
    } catch (e: int) {
        sum += e;
    }

    try {
        throw 1.5;
    } catch (e: number) {
        sum += e * 2;
    }

    try {
        throw "x";
    } catch (e: string) {
        sum += e == "x" ? 10 : 0;
    }

    return sum;
}

function main() {
    assert(untypedNumber() == 2.5, "an untyped catch reads a thrown number");
    assert(untypedInt() == 7, "an untyped catch reads a thrown int");
    assert(untypedString() == "boom", "an untyped catch reads a thrown string");
    assert(untypedClass() == 42, "an untyped catch reads a thrown class instance");
    assert(untypedDerivedAsBase() == 43, "an untyped catch reads a thrown derived instance as either class");
    assert(anyTyped() == 11, "a catch (e: any) reads a thrown int");
    assert(alreadyAny() == 123, "a thrown any is bound as it is");
    assert(numberCatchesInt() == 5.5, "catch (e: number) catches a thrown integer literal");
    assert(typedCatchesUnchanged() == 16, "typed catches still bind their own types");

    print("done.");
}
