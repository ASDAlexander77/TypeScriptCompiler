// a generic function called with nothing to infer its type parameters from takes the
// default, then the constraint (docs/not_impl/not_impl27.ts)

function noParams<T extends {}>() {
    print("noParams");
}

function withDefault<T = number>(): T {
    return 5 as T;
}

function twoDefaults<T = string, U extends number = number>(): string {
    return typeof (0 as U);
}

function main() {
    noParams();
    noParams<string>();
    noParams<{}>();

    assert(withDefault() + 1 == 6, "default");
    assert(withDefault<string>() == "5", "explicit");

    assert(twoDefaults() == "number", "two defaults");

    print("done.");
}
