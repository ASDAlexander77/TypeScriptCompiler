function* f()
{
	yield 1;
	yield 2;	
	yield 3;
}

function main()
{ 
	const v = [...f()];	

	for (const i of v) print(i);

    assert (v.length == 3);

	print("done.");
}