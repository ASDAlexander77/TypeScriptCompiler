// -mm=own rejects, and does not copy: `keep` is private and called only directly, so the
// signature pass tells its callers it keeps its parameter (`__own_params`) and they give theirs up.
// `keep` uses it after storing it; a copy for the store would leak the one the caller gave.
class H {
    s = "";
}

const holder = new H();

function keep(v: string) {
    holder.s = v;
    print(v);
}

function main() {
    keep("x" + 1);
    print(holder.s);
}
