// A C++ destructor's slot is called differently under MSVC (the deleting one, taking flags) and the
// Itanium ABI (the complete one, freeing nothing): a call of it is an error.
@cpp
class Resource {
    @virtual size(): s32;
    @destructor destroy(): void;
}

declare function make_resource(): Resource;

function main() {
    const resource = make_resource();
    resource.destroy();
}
