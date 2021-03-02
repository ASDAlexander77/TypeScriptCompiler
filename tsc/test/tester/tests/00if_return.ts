function main() {    
    assert(test(1) == 3, "failed. 1");
    test2(1);
}

function test(a: number)
{
    if (a == 1)
    {
        return 3;
    }

    return 2;
}

function test2(a: number)
{
    if (a == 1)
    {
        return;
    }

    assert(false, "failed. 2");
}
