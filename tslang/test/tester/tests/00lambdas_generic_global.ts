const convert = <U>(value: U): U => value;

function main()
{
	const v:string = convert("asd");

	print(v);

	assert(v == "asd");

	print("done.");
}