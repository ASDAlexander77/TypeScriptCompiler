function main() {    
    assert(test1() == 10, "failed. 1");
    assert(test2() == 11, "failed. 2");
}

function test1()
{
	for (let i = 0; i < 10; i++)
	{
		print(i);
	}

    return i;
}

function test2()
{
	for (let i = 0; i++ < 10;)
	{
		print(i);
	}

    return i;
}
