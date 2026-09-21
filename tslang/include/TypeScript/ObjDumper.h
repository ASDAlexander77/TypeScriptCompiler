#ifndef OBJDUMPER__H
#define OBJDUMPER__H

#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"

#include "llvm/Support/Allocator.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

using namespace llvm;
using namespace llvm::object;

class Dumper {

public:
  virtual ~Dumper() {}

  virtual void getSymbols(SmallVector<StringRef>&, BumpPtrAllocator &) = 0;
};

class COFFDumper : public Dumper {
public:
  COFFDumper(const object::COFFObjectFile &objFile)
      : coffObj(objFile) {
    is64 = !coffObj.getPE32Header();
  }

  void getSymbols(SmallVector<StringRef>&, BumpPtrAllocator &) override;

private:

  const COFFObjectFile &coffObj;
  bool is64;
};

template <typename ELFT>
class ELFDumper : public Dumper
{
public:
    ELFDumper(const ELFObjectFile<ELFT> &objFile) : elfObjectFile(objFile) {}

    void getSymbols(llvm::SmallVector<StringRef>& symbols, BumpPtrAllocator &stringAllocator) override
    {
        for (auto I = elfObjectFile.symbol_begin(); I != elfObjectFile.symbol_end(); ++I)
        {
            const SymbolRef &symbol = *I;
            auto nameOrError = symbol.getName();
            if (nameOrError)
            {
                auto name = nameOrError.get();
                if (!name.empty())
                {
                    symbols.push_back(StringRef(name).copy(stringAllocator));
                }
            }
        }
    }

private:
    const ELFObjectFile<ELFT> &elfObjectFile;
};

template <class ELFT>
static std::unique_ptr<Dumper> createDumperT(const ELFObjectFile<ELFT> &elfObjectFile)
{
    return std::make_unique<ELFDumper<ELFT>>(elfObjectFile);
}

namespace Dump
{
    void getSymbols(llvm::StringRef, SmallVector<StringRef> &, BumpPtrAllocator &);

    // Whether the binary carries a Boehm collector of its own (linked the static gc library),
    // rather than importing one from gc.dll / its host or having none. Read from the binary itself,
    // so it answers for libraries linked by hand or by an older tslang as well.
    bool containsGarbageCollector(llvm::StringRef);

    // The COFF machine (IMAGE_FILE_MACHINE_*) of an object file, a static library or an import
    // library - for a library, that of its first member that names one - or 0 when there is none
    // to be read. A query: an unreadable or non-COFF file is 0, not an error.
    uint16_t coffMachine(llvm::StringRef);

    // The name a user knows a COFF machine by ("x86", "x64", "arm64"), or its number in hex.
    std::string coffMachineName(uint16_t);

    // The NUL-terminated string that the exported pointer variable `symbol` (a `const char *`)
    // points to in a PE DLL, read from the file without loading it - for a DLL built for another
    // architecture, which this process cannot load. std::nullopt when the file is not a PE image,
    // does not export `symbol`, or the pointer or string lies outside the file's data.
    std::optional<std::string> readExportedCString(llvm::StringRef path, llvm::StringRef symbol);
}

std::unique_ptr<Dumper> createCOFFDumper(const COFFObjectFile &);
std::unique_ptr<Dumper> createELFDumper(const ELFObjectFileBase &);
// TODO: finish it
//std::unique_ptr<Dumper> createMachODumper(const MachOObjectFile &);
//std::unique_ptr<Dumper> createWasmDumper(const WasmObjectFile &);
//std::unique_ptr<Dumper> createXCOFFDumper(const XCOFFObjectFile &);

#endif // OBJDUMPER__H
