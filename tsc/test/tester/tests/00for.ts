function main() {    
    assert(test() == 10, "failed. 1");
}

function test()
{
	for (let i = 0; i < 10; i++)
	{
		print(i);
	}

    return i;
}
