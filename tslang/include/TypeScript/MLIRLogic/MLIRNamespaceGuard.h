#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRNAMESPACEGUARD_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRNAMESPACEGUARD_H_

#include "TypeScript/TypeScriptOps.h"

#include "parser.h"

namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRNamespaceGuard
{
    NamespaceInfo::TypePtr savedNamespace;
    NamespaceInfo::TypePtr &currentNamespace;

  public:
    MLIRNamespaceGuard(NamespaceInfo::TypePtr &currentNamespace) : currentNamespace(currentNamespace)
    {
        savedNamespace = currentNamespace;
    }

    ~MLIRNamespaceGuard()
    {
        currentNamespace = savedNamespace;
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRNAMESPACEGUARD_H_
