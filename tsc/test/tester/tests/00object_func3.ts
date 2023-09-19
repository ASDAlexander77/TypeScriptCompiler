function main() {

    let obj = {
        val: 10,
        add: () => {
            add_();

            function add_() {
                if (++this.val < 15) this.add();
            }
        }
    };

    obj.add();
    print(obj.val);

    assert(obj.val === 15);

    main2();

    print("done.");
}

function main2()
{
    let o2 = { a: 1, 'b': 2, ['c']: 3, d() { }, ['e']: 4 } as const;
    let o9 = { x: 10, foo() { this.x = 20 } } as const;


    o2.d();
    o9.foo();

    assert(o9.x == 20);
}