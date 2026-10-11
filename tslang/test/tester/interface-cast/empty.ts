// An interface without members takes any value, as in TypeScript: this compiles.
interface IEmpty {
}

function f(i: IEmpty) {
    return 1;
}

print(f([1, 2]) + f(5) + f("text"));
