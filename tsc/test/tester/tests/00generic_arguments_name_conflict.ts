type If<C extends boolean, T, F> = C extends true ? T : F;

type And<A extends boolean, B extends boolean> = If<A, B, false>;

type A1 = And<false, false>;  // false

interface A
{
	a: 'A';
}

function main()
{
	print("done.");
}