let a_method3 = 0;
let c_method3 = 0;

class A
{
	method3()
	{	
		a_method3++;
		print("A::method - 3");
	}
}

class B extends A
{
	method()
	{	
		this.method3();
		super.method3();
	}
}

class C extends B
{
	method3()
	{	
		c_method3++;
		print("C::method - 3");
		super.method3();
	}
}

function main()
{
    const c = new C();
    c.method();

    print(a_method3);
    print(c_method3);

    assert(a_method3 == 2);
    assert(c_method3 == 1);

    print("done.");
}