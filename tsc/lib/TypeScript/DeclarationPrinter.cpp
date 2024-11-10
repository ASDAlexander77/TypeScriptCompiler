#include "TypeScript/DeclarationPrinter.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

namespace ts
{

void DeclarationPrinter::print(ClassInfo::TypePtr)
{
    // TODO:
}

} // namespace ts