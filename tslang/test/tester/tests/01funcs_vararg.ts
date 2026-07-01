type Array<T> = T[];

function foo<T>(...args: T[]): T[]
{
	return args;
}

function main()
{
	let b1: { x: boolean }[] = foo({ x: true }, { x: false });
	let b2: boolean[][] = foo([true], [false]);

    main2();

	print("done.");
}

function fn(x: any): void {};
function takeTwo(x: any, y: any): void {};
function withRest(a: any, ...args: Array<any>): void {};

function main2()
{
    let n: number[] = [];

    fn(1) // no error
    takeTwo(1, 2)
    withRest('a', ...n); // no error
    withRest(...n);
}