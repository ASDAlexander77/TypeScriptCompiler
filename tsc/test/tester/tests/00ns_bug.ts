interface IData<T>
{
    data: T;
}

namespace test {
    function f() : IData<string[]> {
        return { data: ["Hello"] };
    }
}

const d = test.f();

print("done.");