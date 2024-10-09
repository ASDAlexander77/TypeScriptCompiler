#ifndef MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_
#define MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/LowerToLLVM/LocationHelper.h"

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

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

    mlir::FusedLoc combineWithCurrentScope(mlir::Location location)
    {
        return combine(location, debugScope.lookup(DEBUG_SCOPE));          
    }

    mlir::NameLoc combineWithName(mlir::Location location, StringRef name)
    {
        return mlir::NameLoc::get(builder.getStringAttr(name), location);
    }

    mlir::FusedLoc combineWithCurrentScopeAndName(mlir::Location location, StringRef name)
    {
        return combineWithCurrentScope(combineWithName(location, name));
    }

    void setFile(StringRef fileName) {
        // TODO: in file location helper
        SmallString<256> FullName(fileName);
        sys::path::remove_filename(FullName);

        auto file = mlir::LLVM::DIFileAttr::get(builder.getContext(), sys::path::filename(fileName), FullName);

        debugScope.insert(FILE_DEBUG_SCOPE, file);
        debugScope.insert(DEBUG_SCOPE, file);
    }

    mlir::Location getCompileUnit(mlir::Location location, StringRef producerName, bool isOptimized) {

        auto file = dyn_cast<mlir::LLVM::DIFileAttr>(debugScope.lookup(FILE_DEBUG_SCOPE));

        unsigned sourceLanguage = llvm::dwarf::DW_LANG_C; 
        auto producer = builder.getStringAttr(producerName);
        auto emissionKind = mlir::LLVM::DIEmissionKind::Full;
        auto compileUnit = mlir::LLVM::DICompileUnitAttr::get(builder.getContext(), sourceLanguage, file, producer, isOptimized, emissionKind);        
       
        debugScope.insert(CU_DEBUG_SCOPE, compileUnit);
        debugScope.insert(DEBUG_SCOPE, file);

        return combine(location, compileUnit);
    }

    mlir::Location getSubprogram(mlir::Location functionLocation, StringRef functionName, mlir::Location functionBlockLocation) {

        auto compileUnitAttr = dyn_cast<mlir::LLVM::DICompileUnitAttr>(debugScope.lookup(CU_DEBUG_SCOPE));
        auto scopeAttr = dyn_cast<mlir::LLVM::DIScopeAttr>(debugScope.lookup(DEBUG_SCOPE));

        auto line = LocationHelper::getLine(functionLocation);
        auto scopeLine = LocationHelper::getLine(functionBlockLocation);

        auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Definition;
        if (compileUnitAttr.getIsOptimized())
        {
            subprogramFlags = subprogramFlags | mlir::LLVM::DISubprogramFlags::Optimized;
        }

        auto type = mlir::LLVM::DISubroutineTypeAttr::get(builder.getContext(), llvm::dwarf::DW_CC_normal, {/*Add Types here*/});

        auto funcNameAttr = builder.getStringAttr(functionName);
        auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
            builder.getContext(), compileUnitAttr, scopeAttr, 
            funcNameAttr, funcNameAttr, 
            compileUnitAttr.getFile(), line, scopeLine, subprogramFlags, type);      

        debugScope.insert(SUBPROGRAM_DEBUG_SCOPE, subprogramAttr);
        debugScope.insert(DEBUG_SCOPE, subprogramAttr);

        return combine(functionLocation, subprogramAttr);
    }

    void setLexicalBlock(mlir::Location blockLocation) {

        auto fileAttr = dyn_cast<mlir::LLVM::DIFileAttr>(debugScope.lookup(FILE_DEBUG_SCOPE));
        auto scopeAttr = dyn_cast<mlir::LLVM::DIScopeAttr>(debugScope.lookup(DEBUG_SCOPE));

        auto [scopeLine, scopeColumn] = LocationHelper::getLineAndColumn(blockLocation);

        auto lexicalBlockAttr = 
            mlir::LLVM::DILexicalBlockAttr::get(
                builder.getContext(), 
                scopeAttr, 
                fileAttr, 
                scopeLine, 
                scopeColumn);      

        debugScope.insert(BLOCK_DEBUG_SCOPE, lexicalBlockAttr);
        debugScope.insert(DEBUG_SCOPE, lexicalBlockAttr);
    }    

private:
    mlir::FusedLoc combine(mlir::Location location, mlir::LLVM::DIScopeAttr scope)
    {
        return mlir::FusedLoc::get(builder.getContext(), {location}, scope);          
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_
