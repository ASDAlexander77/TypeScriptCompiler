function main() {    
    assert(test(1) == 3, "failed. 1");
    test2(1);
    assert(test3(1) == 5, "failed. 3");
    assert(test4(1) == 3, "failed. 4");

    print("done.");
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

function test3(a: number)
{
    if (a == 1)
    {
        a = a + 1;
        a = a + 1;
    }

    a = a + 1;
    a = a + 1;

    return a;
}

function test4(a: number)
{
    if (a == 1)
    {
        a = a + 1;
        a = a + 1;
        return a;
    }

    a = a + 1;
    a = a + 1;

    return a;
}