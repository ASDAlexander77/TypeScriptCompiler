function main()
{
	type A = number;
	const A = "asd";
	const b: A = 1;

	print(b, A);

    assert(A == "asd" && b == 1);
	
	print("done.");
}