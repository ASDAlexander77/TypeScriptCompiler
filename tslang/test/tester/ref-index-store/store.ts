// Indexing a Ref<T> gives the element's reference, as a read does (`Deref(bytes[i])`), so a store
// to it is an error that names the form that stores: `Deref(bytes[i]) = value`.
type Str = [buf: [_0: u64, _1: u64], size: index];

function main() {
    let value: Str = [[0, 0], 0];
    let bytes = <Ref<u8>><Opaque>Ref(value);
    bytes[0] = 65;
}
