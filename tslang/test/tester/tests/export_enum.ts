namespace A
{
    enum Enum1
    {
        A1,
        B1
    }
}

namespace B
{
    export function f(): A.Enum1
    {
        return A.Enum1.A1;
    }
}