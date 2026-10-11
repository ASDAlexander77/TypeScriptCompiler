import './export_os_suffix'

// There is no export_os_suffix.ts: the import takes the target's file, export_os_suffix.win32.ts or
// export_os_suffix.linux.ts, the one compiled and linked beside this (the other one's functions
// would be missing at link time).

function main() {
    assert(O.osFile == "win32" || O.osFile == "linux", "the module's file for the target");
    assert(O.processId() != 0, "and its functions");

    print("done.");
}
