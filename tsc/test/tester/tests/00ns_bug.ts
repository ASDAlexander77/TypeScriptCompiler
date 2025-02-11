interface IData<T>
{
    data: T;
}

namespace ifaces
{
    export interface IData<T>
    {
        data: T;
    }

    export interface IData2
    {
        data: string;
    }    
}

namespace test {
    function f() : ifaces.IData<string[]> {
        return { data: ["Hello"] };
    }

    function f2() : ifaces.IData2 {
        return { data: "Hello" };
    }

    function f3() : IData<string[]> {
        return { data: ["Hello"] };
    }
}

const d = test.f();
const d2 = test.f2();
const d3 = test.f3();

print("done.");