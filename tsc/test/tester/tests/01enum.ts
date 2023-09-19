// @declaration: true

// An enum declaration that specifies a const modifier is a constant enum declaration.
// In a constant enum declaration, all members must have constant values and
// it is an error for a member declaration to specify an expression that isn't classified as a constant enum expression.

const enum E {
    a = 10,
    b = a,
    c = (a+1),
    e,
    d = ~e,
    f = a << 2 >> 1,
    g = a << 2 >>> 1,
    h = a | b,
    i = E.a
}

enum T1 {
    a = "1",
    b = "1" + "2",
    c = "1" + "2" + "3"
}

enum EComp2 {
    A = 'foo'.length, B = 'foo'.length, C = 'foo'.length
}

function main()
{
	assert(E.a === E.i);
	assert(T1.c === "123");
	assert(EComp2.C === 3);
	print("done.");
}