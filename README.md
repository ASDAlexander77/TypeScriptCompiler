# TypeScript Native Compiler
###### Powered by [![LLVM|MLIR](https://llvm.org/img/LLVM-Logo-Derivative-1.png)](https://llvm.org/)

License
-------

TypeScriptCompiler is licensed under the MIT license.

Chat Room
---------

Want to chat with other members of the TypeScriptCompiler community?

[![Join the chat at https://gitter.im/ASDAlexander77/TypeScriptCompiler](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ASDAlexander77/TypeScriptCompiler?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Example

```TypeScript
abstract class Department {
    constructor(public name: string) {}

    printName(): void {
        print("Department name: " + this.name);
    }

    abstract printMeeting(): void; // must be implemented in derived classes
}

class AccountingDepartment extends Department {
    constructor() {
        super("Accounting and Auditing"); // constructors in derived classes must call super()
    }

    printMeeting(): void {
        print("The Accounting Department meets each Monday at 10am.");
    }

    generateReports(): void {
        print("Generating accounting reports...");
    }
}

function main() {
    let department: Department; // ok to create a reference to an abstract type
    department = new AccountingDepartment(); // ok to create and assign a non-abstract subclass
    department.printName();
    department.printMeeting();
    //department.generateReports(); // error: department is not of type AccountingDepartment, cannot access generateReports
}
```

```cmd
tsc --emit=jit example.ts
```

Result
```
Department name: Accounting and Auditing
The Accounting Department meets each Monday at 10am.
```

## Compile as JIT

```cmd
tsc --emit=jit hello.ts
```
File ``hello.ts``

```TypeScript
function main() {
    print("Hello World!");
}
```
Result
```
Hello World!
```

## Compile as Binary Executable

### On Windows
File ``tsc-compile.bat``
```cmd
rem set %LLVM% and %TSCBIN%
set LLVMPATH=%LLVM%\llvm\release\bin
set TSCPATH=%TSCBIN%\tsc\bin
set VCFLD=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\SDK\ScopeCppSDK\vc15
set LIBPATH=%VCFLD%\VC\lib
set SDKPATH=%VCFLD%\SDK\lib
%TSCPATH%\tsc.exe --emit=llvm %FILENAME%.ts 2>%FILENAME%.ll
%LLVMPATH%\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.ll
%LLVMPATH%\lld.exe -flavor link %FILENAME%.o /libpath:"%LIBPATH%" /libpath:"%SDKPATH%" "%LIBPATH%\libcmt.lib" "%LIBPATH%\libvcruntime.lib" "%SDKPATH%\kernel32.lib" "%SDKPATH%\libucrt.lib" "%SDKPATH%\uuid.lib"
```
Compile 
```cmd
tsc-compile.bat hello
```

Run
```
hello.exe
```

Result
```
Hello World!
```

## Build
### On Windows

First, precompile dependencies

```
prepare_3rdParty.bat 
```

To build TSC binaries:

```
cd tsc
config_tsc_debug.bat
build_tsc_debug.bat
```
