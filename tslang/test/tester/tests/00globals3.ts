// `a` is a module-level var: a `var` written inside main() would be local to main (MDN var rules),
// so f() and rrr.t() could not see it.
var a = 10;

function main()
{
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