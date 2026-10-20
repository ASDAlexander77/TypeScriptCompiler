import './export_defaultlib_data'

// Calls a shared library that reads data defined in the default library's DLL. See
// export_defaultlib_data.ts.

function main() {
    const size = L.mapSize(300);
    print("size:", size);
    assert(size == 300, "Map in a shared library");

    assert(L.pi() > 3.14159 && L.pi() < 3.1416, "Math.PI in a shared library");
    assert(L.epsilon() > 0 && L.epsilon() < 1e-15, "Number.EPSILON in a shared library");

    print("done.");
}
