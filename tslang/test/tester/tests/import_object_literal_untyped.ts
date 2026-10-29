import './export_object_literal_untyped'

// Direct read/call of the imported boxed global, no interface cast.
print(A.counterObj.count);
assert(A.counterObj.count == 0);
A.counterObj.inc();
print(A.counterObj.count);
assert(A.counterObj.count == 1);
A.counterObj.inc();
print(A.counterObj.count);
assert(A.counterObj.count == 2);

// Also exercise the interface-cast path (mirrors the previously-fixed
// crash), to confirm both consumption modes work for an untyped export.
var c: A.Counter = <A.Counter>A.counterObj;
c.inc();
print(c.count);
assert(c.count == 3);
// the interface and the import are one object, the exporter's: the importer's variable is its
// boxed pointer, not a copy of what it points to
assert(A.counterObj.count == 3);
assert(A.countOf() == 3);

print("done.");
