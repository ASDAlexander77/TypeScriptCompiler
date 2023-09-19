function main()
{
	let a: { f:() => void } | undefined = undefined;

	print("start");

	a?.f();

	print("done.");
}