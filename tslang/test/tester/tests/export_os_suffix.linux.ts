// The Linux file of the module `import './export_os_suffix'` names (import_os_suffix.ts): an
// import without an extension takes <name>.win32.ts on Windows, <name>.linux.ts on Linux.

declare function getpid(): s32;

namespace O {

    export const osFile = "linux";

    export function processId(): u32 {
        return <u32>getpid();
    }
}
