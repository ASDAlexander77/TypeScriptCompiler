namespace ifaces
{
    export interface IData<T>
    {
        data: T;
    }
}

namespace test {
    function f() : ifaces.IData<string[]> {
        return { data: ["Hello"] };
    }
}

const d = test.f();

print("done.");