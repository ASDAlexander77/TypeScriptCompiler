function f()
{
	let a = 0;
	function test<T>(v: T)
	{
		a = v;
	}

	test<TypeOf<1>>(10);
	print(a);
	assert(a == 10);
}

function main()
{
	f();
	print("done.");
}