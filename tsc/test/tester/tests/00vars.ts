function main() {    
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();

    print("done.");    
}

function test1()
{
    const i = 11;
    assert(i == 11, "Failed. 1");
}

function test2()
{
    let i = 22;
    assert(i == 22, "Failed. 2");
}

function test3()
{
    const i: number = 33;
    assert(i == 33, "Failed. 3");
}

function test4()
{
    let i: number = 44;
    assert(i == 44, "Failed. 4");
}

function test5()
{
    let i: number;
    i = 55;
    assert(i == 55, "Failed. 5");
}

function test6()
{
    let i: number, j: number;
    i = j = 66;
    assert(i == 66, "Failed. 6");
}

function test7()
{
   let [name, age] = ['Mika', 28];

   assert(name == "Mika", "Failed. 7.1");
   assert(age == 28, "Failed. 7.2");
}
