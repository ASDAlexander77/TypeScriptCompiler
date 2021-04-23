function main() {    
    assert(test() == 0, "failed. 1");
}

function test()
{
	let i = 10;
	do
	{
		print (i);
	} while (--i > 0);

    return i;
}
