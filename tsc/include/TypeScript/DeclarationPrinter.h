#ifndef TYPESCRIPT_DECLARATIONPRINTER_H
#define TYPESCRIPT_DECLARATIONPRINTER_H

#include "TypeScript/MLIRLogic/MLIRGenStore.h"

#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;

namespace typescript
{
    class DeclarationPrinter
    {
        raw_ostream &os;

    public:
        DeclarationPrinter(raw_ostream &os) : os(os) {};

        void print(ClassInfo::TypePtr);
    };

} // namespace typescript 

#undef DEBUG_TYPE

#endif // TYPESCRIPT_DECLARATIONPRINTER_H
