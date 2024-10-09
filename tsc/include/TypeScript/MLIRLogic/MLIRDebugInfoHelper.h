#ifndef MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_
#define MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_

#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

namespace mlir_ts = mlir::typescript;

using llvm::StringRef;
using llvm::SmallString;

namespace typescript
{

class MLIRDebugInfoHelper
{
    mlir::OpBuilder &builder;
    llvm::ScopedHashTable<StringRef, mlir::LLVM::DIScopeAttr> &debugScope;

  public:

    MLIRDebugInfoHelper(
        mlir::OpBuilder &builder, llvm::ScopedHashTable<StringRef, mlir::LLVM::DIScopeAttr> &debugScope)
        : builder(builder), debugScope(debugScope)
    {
    }

    mlir::FusedLoc combine(mlir::Location location, mlir::LLVM::DIScopeAttr scope)
    {
        return mlir::FusedLoc::get(builder.getContext(), {location}, scope);          
    }

    mlir::LLVM::DIFileAttr getFile(StringRef fileName) {
        // TODO: in file location helper
        SmallString<256> FullName(fileName);
        sys::path::remove_filename(FullName);

        auto file = mlir::LLVM::DIFileAttr::get(builder.getContext(), sys::path::filename(fileName), FullName);

        debugScope.insert(FILE_DEBUG_SCOPE, file);

        return file;
    }

    mlir::LLVM::DICompileUnitAttr getCompileUnit(StringRef producerName, bool isOptimized) {

        auto file = dyn_cast<mlir::LLVM::DIFileAttr>(debugScope.lookup(FILE_DEBUG_SCOPE));

        unsigned sourceLanguage = llvm::dwarf::DW_LANG_C; 
        auto producer = builder.getStringAttr(producerName);
        auto emissionKind = mlir::LLVM::DIEmissionKind::Full;
        auto compileUnit = mlir::LLVM::DICompileUnitAttr::get(builder.getContext(), sourceLanguage, file, producer, isOptimized, emissionKind);        
       
        debugScope.insert(CU_DEBUG_SCOPE, compileUnit);

        return compileUnit;
    }

};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_
