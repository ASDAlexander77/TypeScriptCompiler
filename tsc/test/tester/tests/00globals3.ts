function main()
{
	var a = 10;

	print(a);

	assert(a == 10);

	f();

	rrr.t();

	print("done.");
}

function f()
{
	assert(a == 10);

	print(a);

	a = 20;

	assert(a == 20);
}

namespace rrr
{
	var r = 30;

	function t()
	{
		print(r);
		print(a);

		assert(a == 20);
		assert(r == 30);
	}
}