#ifndef OBJDUMPER__H
#define OBJDUMPER__H

#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/COFF.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

class Dumper {

public:
  virtual ~Dumper() {}

  virtual void getSymbols(llvm::SmallVector<llvm::StringRef>&) = 0;
};

class COFFDumper : public Dumper {
public:
  COFFDumper(const object::COFFObjectFile &objFile)
      : coffObj(objFile) {
    Is64 = !coffObj.getPE32Header();
  }

  void getSymbols(llvm::SmallVector<llvm::StringRef>&) override;

private:

  const llvm::object::COFFObjectFile &coffObj;
  bool Is64;
};

namespace Dump
{
    llvm::SmallVector<llvm::StringRef> getSymbols(llvm::StringRef filePath);
}

std::unique_ptr<Dumper> createCOFFDumper(const llvm::object::COFFObjectFile &);
//std::unique_ptr<Dumper> createELFDumper(const llvm::object::ELFObjectFileBase &);
//std::unique_ptr<Dumper> createMachODumper(const llvm::object::MachOObjectFile &);
//std::unique_ptr<Dumper> createWasmDumper(const llvm::object::WasmObjectFile &);
//std::unique_ptr<Dumper> createXCOFFDumper(const llvm::object::XCOFFObjectFile &);

#endif // OBJDUMPER__H
