function test1()
{
  let a: [string, number] = ["asd", 1.0];
  print(a[0], a[1]);  


  const b: [string, number] = ["asd", 1.0];
  print(b[0], b[1]);  

  const c = ["asd", 1.0];
  print(c[0], c[1]);  
}

function test2()
{
    const d: [ [number, string], number ] = [ [ 1.0, "asd" ], 2.0 ];

    const v1 = d[0];
    const v2 = v1[0];
    print (v2);
  
    print (d[0][1]);  
}

function main()
{
    test1();
    test2();
}