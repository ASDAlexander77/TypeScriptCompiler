let called = false;

function may_throw(a = 0)
{
	if (a > 100) throw 1;
}

function func()
{
	try
	{
		print("In try");
		may_throw(10);
		return;
	}
	finally
	{
		print("finally");
        called = true;
	}

	print("end");
}

function main()
{
	func();
    assert(called, "finally is not called");
	print("done.");
}