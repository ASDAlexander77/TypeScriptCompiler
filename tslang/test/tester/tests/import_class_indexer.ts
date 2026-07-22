import './export_class_indexer'

class UpperStorage extends M.Storage {
    [index: number]: string;

    get(index: number): string {
        return super[index] + "!";
    }

    set(index: number, value: string) {
        super[index] = "U:" + value;
    }
}

function main() {
    const s = new UpperStorage();
    s[0] = "hi";
    assert(s[0] == "U:hi!");

    print("done.");
}
