function gen<T>(t: T)
{
    if (typeof t == "array")
    {
        return t.length.toString();
    }

    return "int";
}

const v1 = gen<i32>(23);
print(v1);

const v2 = gen<string[]>(null);
print(v2);

print("done.");
