#ifndef TYPESCRIPT_DECLARATIONPRINTER_H
#define TYPESCRIPT_DECLARATIONPRINTER_H

#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;

namespace ts
{
    class DeclarationPrinter
    {
        raw_ostream &os;

    public:
        DeclarationPrinter(raw_ostream &os);
    };

} // namespace ts 

#endif // TYPESCRIPT_DECLARATIONPRINTER_H
