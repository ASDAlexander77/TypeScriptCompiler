#ifndef TYPESCRIPT_LIBRARY_DECLARATIONS_H
#define TYPESCRIPT_LIBRARY_DECLARATIONS_H

// What a module importing a tslang shared library needs to know about it, kept in a file beside the
// library: `<library stem>.d.ts` (`foo.dll` -> `foo.d.ts`, `libfoo.so` -> `libfoo.d.ts`). It used
// to be exported from the library itself (`__decls_<module>` and the rest), which a compiler can
// read only from a binary it can load or parse: an Android `.so` could not be imported on Windows.
//
// The file holds one block per module the library was built from - `--emit=obj` writes one beside
// each object, and a library linked from several objects gets them one after another (test-runner
// concatenates them, `--emit=dll` merges those of the objects it links). A block is:
//
//   //@tslang-module <moduleSymbolSuffix> model=<gc|rc|none|own> [gctors]
//   //@tslang-declarations
//   ...the module's exported declarations, ambient (parsed as a .d.ts)...
//   //@tslang-generic
//   ...its generic classes, parsed as source: their specializations are compiled by the importer...
//   //@tslang-own-facts
//   // <name of an exported function that destroys nothing a caller can reach>
//
// Every section but the first line is optional. The file is a valid .d.ts, but it is never read as
// one: the two kinds of declarations are parsed differently, module by module.

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace typescript
{

// the first line of every block, and so of the file
#define LIBRARY_DECLARATIONS_FILE_MARKER "// tslang: the declarations of a shared library, read by `import`"

struct LibraryModuleDeclarations
{
    // moduleSymbolSuffix of the module's source: the "<file>_<hash>" in `__decls_<file>_<hash>`
    std::string suffix;
    std::string memoryModel;
    // the library exports __mlir_gctors (--gctors-as-method), which a program loading it calls
    bool hasGCtors = false;
    std::string declarations;
    std::string genericDeclarations;
    // names, one per line
    std::string ownFacts;
};

// `foo.dll` -> `foo.d.ts`, `libfoo.so` -> `libfoo.d.ts`, `foo.obj` -> `foo.d.ts`
std::string libraryDeclarationsPath(llvm::StringRef binaryPath);

std::string printLibraryDeclarations(const std::vector<LibraryModuleDeclarations> &modules);

// nullopt: not a file of this kind (no marker on its first line)
std::optional<std::vector<LibraryModuleDeclarations>> parseLibraryDeclarations(llvm::StringRef text);

// the file's text is one of these (it starts with the marker)
bool isLibraryDeclarationsText(llvm::StringRef text);

} // namespace typescript

#endif // TYPESCRIPT_LIBRARY_DECLARATIONS_H
