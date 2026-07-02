// @strict-null false
function cast<T>(a: any) : T
{
	if (typeof a == "number") return <T>a;
	if (typeof a == "string") return <T>a;
	if (typeof a == "i32") return <T>a;
	if (typeof a == "s32") return <T>a;
	if (typeof a == "u32") return <T>a;
	if (typeof a == "class") if (a instanceof T) return <T>a;
	return <T>null;
}

function eq<A, B>(a: A, b: B) { return a == b as any as A }

function eq2<A, B>(a: A, b: B) { 
	return a == cast<A>(b as any); 
}

function main() {

    assert(eq("2", 2), "2")
    assert(eq(2, "2"), "2'")
    //assert(!eq("null", null), "=1") // TODO: null ref here
    //assert(!eq(null, "null"), "=2")
    assert(!eq("2", 3), "=3")

    assert(eq2("2", 2), "2")
    assert(eq2(2, "2"), "2'")
    //assert(!eq2("null", null), "=1") // TODO: null ref here
    //assert(!eq2(null, "null"), "=2")
    assert(!eq2("2", 3), "=3")

    print("done.");
}
