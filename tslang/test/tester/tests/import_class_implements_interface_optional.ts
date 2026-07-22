import './export_class_implements_interface_optional'

function main() {
    const withOpt: M.Shape = new M.WithOpt();
    assert(withOpt.base == 1);
    assert(withOpt.opt == 5);

    const withoutOpt: M.Shape = new M.WithoutOpt();
    assert(withoutOpt.base == 2);
    assert(withoutOpt.opt == undefined);

    print("done.");
}
