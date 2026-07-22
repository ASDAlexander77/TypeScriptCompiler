import './export_class_implements_interface_multilevel'

function main() {
    const obj = new M.Impl();
    assert(obj.describe() == "base=1,derived=2");

    const asDerived: M.Derived = obj;
    assert(asDerived.base == 1);
    assert(asDerived.derived == 2);
    assert(asDerived.describe() == "base=1,derived=2");

    const asBase: M.Base = obj;
    assert(asBase.base == 1);
    assert(asBase.describe() == "base=1,derived=2");

    print("done.");
}
