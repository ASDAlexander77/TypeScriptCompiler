function t(a: { name: string })
{
   print(a.name);
}

function main() {
    const object = { a: 1, b: 2, c: 3 };

    print(object.a, object.b, object.c);

    let object2 = { a: 1, b: 2, c: 3 };

    print(object2.a, object2.b, object2.c);

    object2.a = 10;
    object2.b = 20;
    object2.c = 20;

    print(object2.a, object2.b, object2.c);

    const a = 'foo';
    const b = 42;
    const c = {};
    const object3 = { a: a, b: b, c: c };

    print(object3.a, object3.b, object3.c);

    const object4 = { a, b, c };

    print(object4.a);

    const object5 : { name: string } = { name: "foo" };

    print(object5.name);

    t(object5);

    print("done.");    
}
