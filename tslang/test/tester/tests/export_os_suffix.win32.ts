// The Windows file of the module `import './export_os_suffix'` names (import_os_suffix.ts): an
// import without an extension takes <name>.win32.ts on Windows, <name>.linux.ts on Linux.

// __stdcall, as the Windows API is: on 32-bit Windows kernel32.lib names it _GetCurrentProcessId@0
@stdcall
declare function GetCurrentProcessId(): u32;

namespace O {

    export const osFile = "win32";

    export function processId(): u32 {
        return GetCurrentProcessId();
    }
}
