function* f()
{
	yield 1;
	yield 2;	
	yield 3;
}

interface IData
{
	a: number;
	b: string;
	c?: number;
}

class S
{
	f = 12;
	d = "asd";
}

function main()
{ 
	const v = [...f()];	

	for (const i of v) print(i);

    assert (v.length == 3);

	let data: IData = { a: 10, b: "Hello" };
	let o = { ...data };

	print(o.a, o.b, o.c || 20);

    assert(o.a === 10);
    assert(o.b === "Hello");
    assert((o.c || 20) === 20);

	let o2 = { ...new S() };

	print(o2.f, o2.d);

    assert(o2.f === 12);
    assert(o2.d === "asd");
    
	print("done.");
}