import './export_interface_indexer'

class Impl {
    #val: string = "";

    [index: number]: string;

    get(index: number): string {
        return this.#val;
    }

    set(index: number, value: string) {
        this.#val = value;
    }
}

function main() {
    const impl = new Impl();
    const s: M.Storage = impl;

    s[0] = "hi";
    assert(s[0] == "hi");

    print("done.");
}
