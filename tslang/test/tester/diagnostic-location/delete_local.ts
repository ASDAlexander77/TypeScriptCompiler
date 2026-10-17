// `delete` of a local holding a number compiled with no error: the error was made and dropped.

class C {}

function main() {
    const c = new C();
    delete c;
    const n = 5;
    delete n;
}
