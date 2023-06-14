function main()
{
	var a = 10;

	print(a);

	assert(a == 10);

	f();

	print("done.");
}

function f()
{
	assert(a == 10);

	print(a);

	a = 20;

	assert(a == 20);
}