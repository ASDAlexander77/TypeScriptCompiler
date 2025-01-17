function get(i: number): string | null
{
    return i % 2 ? "test" : null;
}

let a: string | null = null;

a = get(1);

print(a === "test");

let r = false;
while (a !== null) {
    r = true;
    assert(a === "test");
    print(a, a === "test");
    break;
}   

assert(r);

print("done.")