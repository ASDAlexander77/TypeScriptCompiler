// regression / coverage test: binary operations between operands of DIFFERENT static
// types (string+number, number+string, string+boolean, boolean+string, number+enum,
// any-typed mixed comparisons). These exercise the coercion rules in
// adjustTypesForBinaryOp / cast() (MLIRGenImpl.h, MLIRGenCast.cpp) that aren't covered
// by 00bool_arith_ops.ts (bool+number), 00strings.ts (string+number concat/compare) or
// arithmeticOperatorWithEnum.ts (enum+number).

function stringNumber() {
    // string + number -> string concat, number coerced via toString
    assert("val=" + 42 == "val=42");
    assert(42 + "=val" == "42=val");
    assert("pi=" + 3.5 == "pi=3.5");
    assert(-1 + "x" == "-1x");

    // compound assignment across types
    let s = "n=";
    s += 7;
    assert(s == "n=7");

    // relational compare: number coerced to string, then lexicographic compare
    assert("9" + 0 > "8" + 9); // "90" > "89"
    assert(!("2" + 0 < "1" + 9)); // "20" < "19" is false
}

function stringBoolean() {
    // string + boolean -> string concat, boolean coerced via toString
    assert("flag=" + true == "flag=true");
    assert("flag=" + false == "flag=false");
    assert(true + "!" == "true!");
    assert(false + "!" == "false!");

    let s = "b=";
    s += true;
    assert(s == "b=true");
}

function numberBooleanCompare() {
    // number vs boolean relational/equality (boolean coerces to number: true->1, false->0)
    assert(1 == true);
    assert(0 == false);
    assert(1 != false);
    assert(2 > true);
    assert(!(0 > false));
    assert(true >= 1);
    assert(false <= 0);

    // strict equality does NOT coerce across types
    assert(!(1 === true));
    assert(!(0 === false));
}

function enumNumber() {
    enum Level { Low, Medium, High }

    let l = Level.Medium;
    assert(l == 1);
    assert(l + 1 == 2);
    assert(l * 2 == 2);
    assert(l < Level.High);
    assert("level=" + l == "level=1");
}

function anyMixed() {
    let a: any = "abc";
    let b: any = "abc";
    assert(a == b);
    assert(a === b);

    // loose equality across differing `any` payload kinds coerces like JS:
    // number<->string, boolean<->number, boolean<->string
    a = 5;
    b = "5";
    assert(a == b);
    assert(b == a);
    assert(!(a === b));   // strict equality checks type too, this direction works

    a = true;
    b = 1;
    assert(a == b);
    assert(b == a);

    a = false;
    b = 0;
    assert(a == b);

    a = true;
    b = "true";
    assert(a == b);

    a = 5;
    b = "6";
    assert(a != b);
}

function main() {
    stringNumber();
    stringBoolean();
    numberBooleanCompare();
    enumNumber();
    anyMixed();

    print("done.");
}
