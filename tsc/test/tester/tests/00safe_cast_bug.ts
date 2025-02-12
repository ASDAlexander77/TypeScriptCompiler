let array1: string | number | null = 0;

let calls = 0;
function exec(str1: string): string | null
{
    calls ++;
    if (calls <= 1) return "asd";
    return null;
}

while ((array1 = exec("str1")) !== null) {
    if (array1 == null) print("ok");
    print("while loop");

    if (calls > 3) assert(false);
}

assert(calls == 2);

print("calls = ", calls);
print("done.");