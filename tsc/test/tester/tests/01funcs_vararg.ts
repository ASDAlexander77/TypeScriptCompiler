function foo<T>(...args: T[]): T[]
{
	return args;
}

function main()
{
	let b1: { x: boolean }[] = foo({ x: true }, { x: false });
	let b2: boolean[][] = foo([true], [false]);
	print("done.");
}
