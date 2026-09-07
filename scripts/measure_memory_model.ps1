# Builds a native executable per memory model, runs it, and reports peak working set.
#
# Recreated from the description in docs/reference-counting-evaluation.md section 9.52, with one
# change. That section samples PeakWorkingSet64 in a spin loop and warns never to sleep in it,
# because the counter reads zero once the process has exited. The warning is right and the
# technique is still a race: a program that finishes before the first sample reads 0.0 MB, which
# is what it did on the first run here.
#
# The kernel keeps the peak for as long as a handle to the process is open, exited or not, so
# GetProcessMemoryInfo answers after WaitForExit with no sampling at all. Start-Process -PassThru
# holds that handle. No loop, no race, and it costs nothing.
#
# What section 9.52 insists on and is kept: the exit code is printed. A crashed process reports a
# small number and looks like a win - section 9.43's famous 2.6 MB was a process that had died.
param(
    [Parameter(Mandatory=$true)][string]$Source,
    [string[]]$Models = @("gc","rc","none"),
    [string]$Opt = "--opt --opt_level=3"
)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class PeakWs {
    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_MEMORY_COUNTERS {
        public uint cb;
        public uint PageFaultCount;
        public IntPtr PeakWorkingSetSize;
        public IntPtr WorkingSetSize;
        public IntPtr QuotaPeakPagedPoolUsage;
        public IntPtr QuotaPagedPoolUsage;
        public IntPtr QuotaPeakNonPagedPoolUsage;
        public IntPtr QuotaNonPagedPoolUsage;
        public IntPtr PagefileUsage;
        public IntPtr PeakPagefileUsage;
    }
    [DllImport("psapi.dll", SetLastError=true)]
    static extern bool GetProcessMemoryInfo(IntPtr h, out PROCESS_MEMORY_COUNTERS c, uint size);

    // Readable after the process has exited, for as long as the handle is open - which is the
    // whole reason this exists instead of a sampling loop.
    public static long Of(IntPtr handle) {
        PROCESS_MEMORY_COUNTERS c;
        c.cb = 0;
        if (!GetProcessMemoryInfo(handle, out c, (uint)Marshal.SizeOf(typeof(PROCESS_MEMORY_COUNTERS))))
            return -1;
        return (long)c.PeakWorkingSetSize;
    }
}
"@

$bin  = "I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/bin"
$lib  = "I:/TypeScriptCompiler/__build/tslang/windows-msbuild-2026-release/lib"
$lld  = "I:/TypeScriptCompiler/tslang/../3rdParty/llvm/x64/release/bin"
$llvmlib = "I:/TypeScriptCompiler/tslang/../3rdParty/llvm/x64/release/lib"
$gclib = "I:/TypeScriptCompiler/3rdParty/gc/x64/release/lib"
$vclib = "C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/MSVC/14.51.36231/lib/x64"
$sdk   = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.28000.0/um/x64"
$ucrt  = "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.28000.0/ucrt/x64"

$libs = "libcmt.lib libvcruntime.lib libucrt.lib ntdll.lib TypeScriptAsyncRuntime.lib gc.lib LLVMSupport.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib"

$stem = [System.IO.Path]::GetFileNameWithoutExtension($Source)
$work = Join-Path $env:TEMP "measure-$stem"
New-Item -ItemType Directory -Force -Path $work | Out-Null

foreach ($m in $Models) {
    $obj = Join-Path $work "$stem-$m.obj"
    $exe = Join-Path $work "$stem-$m.exe"

    $compile = & "$bin/tslang.exe" --emit=obj $Opt.Split(' ') --no-default-lib "-mm=$m" $Source "-o=$obj" 2>&1
    if ($LASTEXITCODE -ne 0) { "{0,-5} COMPILE FAILED ({1})" -f $m, $LASTEXITCODE; $compile | Select-Object -Last 3; continue }

    $link = & "$lld/lld.exe" -flavor link $obj "/out:$exe" $libs.Split(' ') `
        "/libpath:$gclib" "/libpath:$llvmlib" "/libpath:$lib" "/libpath:$vclib" "/libpath:$sdk" "/libpath:$ucrt" 2>&1
    if ($LASTEXITCODE -ne 0) { "{0,-5} LINK FAILED ({1})" -f $m, $LASTEXITCODE; $link | Select-Object -Last 3; continue }

    $p = Start-Process -FilePath $exe -PassThru -NoNewWindow -RedirectStandardOutput "$work\$stem-$m.out" -RedirectStandardError "$work\$stem-$m.err"
    $p.WaitForExit()
    $peak = [PeakWs]::Of($p.Handle)
    "{0,-5} peak {1,8:N1} MB   exit {2}" -f $m, ($peak/1MB), $p.ExitCode
}
