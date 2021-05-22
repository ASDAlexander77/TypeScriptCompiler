function main() {    
    assert(test() == 0, "failed. 1");

    print("done.");    
}

function test()
{
	let i = 0;
	while (i > 0)
	{
		print (i--);
	}

    return i;
}
