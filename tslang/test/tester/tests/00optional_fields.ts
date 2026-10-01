// A `?` field is `T | undefined`: absent, it reads as undefined - not as T's zero - read directly, through
// an interface, after `{}`, and spread
type Options = { label?: string, size?: number, extra?: any };

interface IOptions {
    label?: string;
    size?: number;
}

interface ITagged extends IOptions {
    tag?: string;
}

type Tagged = { label?: string, tag?: string, kind: string };

function typeLiteral() {
    const none: Options = {};
    assert(none.label === undefined, "{}.label");
    assert(none.size === undefined, "{}.size");
    assert(none.extra === undefined, "{}.extra");

    const some: Options = { label: "x" };
    assert(some.label == "x", "label");
    assert(some.size === undefined, "absent size");
    assert((some.size ?? -1) == -1, "size ?? -1");

    let changed: Options = {};
    changed.size = 2.5;
    assert(changed.size == 2.5, "set size");
    changed.size = undefined;
    assert(changed.size === undefined, "unset size");

    if (some.label) {
        assert(some.label.length == 1, "narrowed label");
    }
}

function readThrough(o: IOptions, label: string, size: number) {
    assert(label == "-" ? o.label === undefined : o.label == label, "interface label");
    assert(size < 0 ? o.size === undefined : o.size == size, "interface size");
}

function throughInterface() {
    const empty: IOptions = {};
    assert(empty.label === undefined, "interface {}.label");
    assert(empty.size === undefined, "interface {}.size");

    readThrough({}, "-", -1);
    readThrough({ label: "l" }, "l", -1);

    // a tuple with `?` fields: the interface reads the value when it is there, undefined when it is not
    const none: Options = {};
    readThrough(none, "-", -1);
    const some: Options = { label: "v", size: 1.5 };
    readThrough(some, "v", 1.5);

    const viaInterface: IOptions = some;
    viaInterface.label = "w";
    assert(viaInterface.label == "w", "write through interface");

    const all: IOptions[] = [some, none, { size: 2.5 }];
    assert(all[1].label === undefined && all[2].size == 2.5, "array of interfaces");
}

function extended() {
    const withTag: Tagged = { kind: "k", tag: "t" };
    const t: ITagged = withTag;
    assert(t.tag == "t", "extends: own member");
    assert(t.label === undefined, "extends: inherited absent member");

    const withLabel: Tagged = { kind: "k", label: "l" };
    const l: ITagged = withLabel;
    assert(l.label == "l", "extends: inherited member");
    assert(l.tag === undefined, "extends: own absent member");
}

interface IWithMethod {
    label?: string;
    name(): string;
}

function withMethod() {
    const o = { label: undefined as string | undefined, name() { return "n"; } };
    const m: IWithMethod = o;
    assert(m.label === undefined, "method-bearing object: absent");
    assert(m.name() == "n", "method-bearing object: method");
}

type Descriptor<D> = { data?: D };

function spread() {
    const d: Descriptor<{ x: number }> = { data: { x: 1.5 } };
    const copy = { ...d.data };
    assert(copy.x == 1.5, "spread optional");
}

function main() {
    typeLiteral();
    throughInterface();
    extended();
    withMethod();
    spread();
    print("done.");
}
