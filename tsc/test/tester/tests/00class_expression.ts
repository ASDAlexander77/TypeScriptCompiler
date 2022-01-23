// TODO: bug.  because x class and class in foo are different types, static fields are not copied
// to implement it, you can store all static fields into VTable, and access them via class reference but when you access static via class_storage you can do as for C++
function foo(x = class { static prop: string }): string {
    return undefined;
}

function foo2(x = class { static prop: string; static func(); }) 
{
    x.func();
}

function main() {
    foo(class { static prop = "hello" }).length;

    foo2(class { 
		static prop = "asdasd";
		static func() 
		{ 
			print("Hello World 2", this.prop); 
		} 
	});

    main2();

    print("done.");
}

function main2() {
    const a = class { static prop = "hello" };

    function f(p: typeof a) {
        print(p.prop);
    }

    f(a);
}