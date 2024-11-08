#ifndef OBJDUMPER__H
#define OBJDUMPER__H

#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace llvm::object;

class Dumper {

public:
  virtual ~Dumper() {}

  virtual void getSymbols(SmallVector<StringRef>&) = 0;
};

class COFFDumper : public Dumper {
public:
  COFFDumper(const object::COFFObjectFile &objFile)
      : coffObj(objFile) {
    is64 = !coffObj.getPE32Header();
  }

  void getSymbols(SmallVector<StringRef>&) override;

private:

  const COFFObjectFile &coffObj;
  bool is64;
};

template <typename ELFT>
class ELFDumper : public Dumper
{
public:
    ELFDumper(const ELFObjectFile<ELFT> &objFile) : elfObjectFile(objFile) {}

    void getSymbols(llvm::SmallVector<llvm::StringRef>& symbols) override
    {
        for (auto I = elfObjectFile.symbol_begin(); I != elfObjectFile.symbol_end(); ++I)
        {
            const SymbolRef &symbol = *I;
            auto nameOrError = symbol.getName();
            if (nameOrError)
            {
                symbols.push_back(nameOrError.get());
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
    llvm::SmallVector<llvm::StringRef> getSymbols(llvm::StringRef filePath);
}

std::unique_ptr<Dumper> createCOFFDumper(const COFFObjectFile &);
std::unique_ptr<Dumper> createELFDumper(const ELFObjectFileBase &);
// TODO: finish it
//std::unique_ptr<Dumper> createMachODumper(const MachOObjectFile &);
//std::unique_ptr<Dumper> createWasmDumper(const WasmObjectFile &);
//std::unique_ptr<Dumper> createXCOFFDumper(const XCOFFObjectFile &);

#endif // OBJDUMPER__H
