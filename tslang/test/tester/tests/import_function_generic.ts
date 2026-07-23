import './export_function_generic'

function main() {
    assert(M.identity<number>(42) == 42);
    assert(M.identity<string>("hi") == "hi");
    assert(M.pair<number, string>(1, "one") == "1-one");

    print("done.");
}
