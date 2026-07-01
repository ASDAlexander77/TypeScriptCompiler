const enum TestType { foo, bar }
type TestTypeStr = keyof typeof TestType;

function f1(f: TestType) { }
function f2(f: TestTypeStr) { }

function main() {
    f1(TestType.foo)
    f1(TestType.bar)
    f2('foo')
    f2('bar')

    print("done.");
}