function isArray<T extends unknown[]>(value: T): value is T {
    return true;
}

function gen<T>(t: T)
{
    if (isArray(t))
    {
        return t.length.toString();
    }

    return "int";
}

const v1 = gen<i32>(23);
print(v1);
assert(v1 == "int");

const v2 = gen<string[]>(null);
print(v2);
assert(v2 == "0");

print("done.");
