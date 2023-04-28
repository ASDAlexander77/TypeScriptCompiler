function eqOp() {
    print("eqOp");
    let x = 12;
    assert((x += 10) == 22, "Y0");
    assert(x == 22, "Y1");
    x /= 2;
    assert(x == 11, "Y2");

    let s = "fo" + 1;
    let t = "ba" + 2;
    s += t;
    assert(s == "fo1b" + "a2", "fb");
}

function eqOpString() {
    print("eqOpStr");
    let x = "fo";
    assert((x += "ba") == "foba", "SY0");
    assert(x == "foba", "SY1");
}

function main() {
    eqOp();
    eqOpString();
    eqG();
    print("done.");
}

// TODO: finish "as"
//function eq<A, B>(a: A, b: B) { return a == b as any as A }

function __as<T>(a: any) : T
{
    // this is "safe cast" logic
	if (typeof a == "number") return <T>a;
	if (typeof a == "string") return <T>a;
	if (typeof a == "i32") return <T>a;
	if (typeof a == "class") if (s instanceof T) return <T>a;
	return <T>null;
}

function eq<A, B>(a: A, b: B) { return a == __as<A>(b as any); }

function eqG() {
    assert(eq("2", 2), "2")
    assert(eq(2, "2"), "2'")
    //assert(!eq("null", null), "=1") // TODO: null ref here
    //assert(!eq(null, "null"), "=2") // TODO: null to "null"
    assert(!eq("2", 3), "=3")
}
