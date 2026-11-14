// TSLANG_ISSUES 48: a @final accessor cannot be overridden
class Base {
    @final get factor(): number { return 1; }
}

class App extends Base {
    get factor(): number { return 2; }
}

function main() { print(new App().factor); }
