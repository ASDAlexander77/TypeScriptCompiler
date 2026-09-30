// A call to the collector's own API, as lib.native.d.ts declares it. Only a `gc` program has the
// collector: under any other model this is a compile error, not an unresolved symbol at link time.
declare function GC_gcollect(): void;

function main() {
    GC_gcollect();
    print("done.");
}
