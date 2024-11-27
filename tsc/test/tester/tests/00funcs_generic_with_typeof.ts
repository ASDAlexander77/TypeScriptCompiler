function gen<T>(t: T)
{
    if (typeof t == "string")
    {
        return t;
    }

    return "no";
}

const v1 = gen<string>("test");
assert(v1 == "test")
print(v1);

const v2 = gen<string[]>(null);
assert(v2 == "no")
print(v2);

const v3 = gen<string[]>();
assert(v3 == "no")
print(v3);

print("done.");
