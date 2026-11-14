// TSLANG_ISSUES 48: a @final method cannot be overridden
class Base {
    @final factor(): number { return 1; }
}

class App extends Base {
    factor(): number { return 2; }
}

function main() { print(new App().factor()); }
