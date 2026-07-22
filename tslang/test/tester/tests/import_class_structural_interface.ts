import './export_class_structural_interface'

// declared entirely in the importer, also with no `implements` clause -
// structurally cast to the EXPORTER's interfaces, and omits the optional
// `tag` field (unlike M.Widget, which provides it)
class LocalWidget {
    id: number = 7;

    describe(): string {
        return `Local#${this.id}`;
    }
}

function main() {
    const w = new M.Widget();

    const asPrintable: M.Printable = w;
    assert(asPrintable.describe() == "Widget#42");
    assert(asPrintable.tag == "widget");

    const asPrintableWithId: M.PrintableWithId = w;
    assert(asPrintableWithId.id == 42);
    assert(asPrintableWithId.describe() == "Widget#42");
    assert(asPrintableWithId.tag == "widget");

    const lw = new LocalWidget();

    const asLocalPrintable: M.Printable = lw;
    assert(asLocalPrintable.describe() == "Local#7");
    assert(asLocalPrintable.tag == undefined);

    const asLocalPrintableWithId: M.PrintableWithId = lw;
    assert(asLocalPrintableWithId.id == 7);
    assert(asLocalPrintableWithId.describe() == "Local#7");
    assert(asLocalPrintableWithId.tag == undefined);

    print("done.");
}
