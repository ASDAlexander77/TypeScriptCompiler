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

    mlir::Location stripMetadata(mlir::Location location) 
    {
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(location))
        {
            if (fusedLoc.getMetadata()) 
            {
                return mlir::FusedLoc::get(fusedLoc.getContext(), fusedLoc.getLocations());
            }
        }

        return location;        
    }

    mlir::Location combineWithFileScope(mlir::Location location)
    {
        if (auto fileScope = dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(debugScope.lookup(FILE_DEBUG_SCOPE)))
        {
            return combine(location, fileScope);          
        }

        return location;
    }

    mlir::Location combineWithCompileUnitScope(mlir::Location location)
    {
        if (auto cuScope = dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(debugScope.lookup(CU_DEBUG_SCOPE)))
        {
            return combine(location, cuScope);          
        }

        return location;
    }

    mlir::Location combineWithCurrentScope(mlir::Location location)
    {
        if (auto localScope = dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(debugScope.lookup(DEBUG_SCOPE)))
        {
            return combine(location, localScope);          
        }

        return location;
    }

    mlir::Location combineWithCurrentLexicalBlockScope(mlir::Location location)
    {
        if (auto lexicalBlockScope = dyn_cast_or_null<mlir::LLVM::DILexicalBlockAttr>(debugScope.lookup(DEBUG_SCOPE)))
        {
            return combine(location, lexicalBlockScope);          
        }

        return location;
    }

    mlir::NameLoc combineWithName(mlir::Location location, StringRef name)
    {
        return mlir::NameLoc::get(builder.getStringAttr(name), location);
    }

    mlir::Location combineWithCurrentScopeAndName(mlir::Location location, StringRef name)
    {
        return combineWithCurrentScope(combineWithName(location, name));
    }

    mlir::Location combineWithFileScopeAndName(mlir::Location location, StringRef name)
    {
        return combineWithFileScope(combineWithName(location, name));
    }    

    mlir::Location combineWithCompileUnitScopeAndName(mlir::Location location, StringRef name)
    {
        return combineWithCompileUnitScope(combineWithName(location, name));
    }    

    void clearDebugScope() 
    {
        debugScope.insert(DEBUG_SCOPE, mlir::LLVM::DIScopeAttr());
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

        if (auto file = dyn_cast_or_null<mlir::LLVM::DIFileAttr>(debugScope.lookup(FILE_DEBUG_SCOPE)))
        {
            unsigned sourceLanguage = llvm::dwarf::DW_LANG_Assembly; 
            auto producer = builder.getStringAttr(producerName);
            auto emissionKind = mlir::LLVM::DIEmissionKind::Full;
            auto namedTable = mlir::LLVM::DINameTableKind::Default;
            auto compileUnit = mlir::LLVM::DICompileUnitAttr::get(
                builder.getContext(), DistinctAttr::create(builder.getUnitAttr()), sourceLanguage, file, producer, isOptimized, emissionKind, namedTable);        
        
            debugScope.insert(CU_DEBUG_SCOPE, compileUnit);
            debugScope.insert(DEBUG_SCOPE, compileUnit);

            return combine(location, compileUnit);
        }

        return location;
    }

    mlir::Location getSubprogram(mlir::Location functionLocation, StringRef functionName, StringRef linkageName, mlir::Location functionBlockLocation, bool cuScope) {

        if (auto compileUnitAttr = dyn_cast_or_null<mlir::LLVM::DICompileUnitAttr>(debugScope.lookup(CU_DEBUG_SCOPE)))
        {
            if (auto scopeAttr = dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(debugScope.lookup(DEBUG_SCOPE)))
            {
                LocationHelper lh(builder.getContext());
                auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(functionLocation);
                auto [line, column] = lineAndColumn;
                auto [scopeFile, scopeLineAndColumn] = lh.getLineAndColumnAndFile(functionBlockLocation);
                auto [scopeLine, scopeColumn] = scopeLineAndColumn;

                // if (isa<mlir::LLVM::DILexicalBlockAttr>(scopeAttr))
                // {
                //     auto file = dyn_cast<mlir::LLVM::DIFileAttr>(debugScope.lookup(FILE_DEBUG_SCOPE));

                //     // create new scope: DICompositeType
                //     //unsigned tag, StringAttr name, DIFileAttr file, uint32_t line, DIScopeAttr scope, 
                //     //DITypeAttr baseType, DIFlags flags, uint64_t sizeInBits, uint64_t alignInBits, ::llvm::ArrayRef<DINodeAttr> elements
                //     auto compositeTypeAttr = mlir::LLVM::DICompositeTypeAttr::get(
                //         builder.getContext(), llvm::dwarf::DW_TAG_class_type, builder.getStringAttr("nested_function"),
                //         file, line, scopeAttr, mlir::LLVM::DITypeAttr(), mlir::LLVM::DIFlags::TypePassByValue | mlir::LLVM::DIFlags::NonTrivial, 0/*sizeInBits*/, 
                //         8/*alignInBits*/, {/*Add elements here*/});

                //     //debugScope.insert(DEBUG_SCOPE, compositeTypeAttr);
                // }

                auto subprogramFlags = mlir::LLVM::DISubprogramFlags::Definition;
                if (compileUnitAttr.getIsOptimized())
                {
                    subprogramFlags = subprogramFlags | mlir::LLVM::DISubprogramFlags::Optimized;
                }

                // add return types
                auto type = mlir::LLVM::DISubroutineTypeAttr::get(builder.getContext(), llvm::dwarf::DW_CC_normal, {/*Add Types here*/});

                auto funcNameAttr = builder.getStringAttr(functionName);
                auto linkageNameAttr = builder.getStringAttr(linkageName);
                auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
                    builder.getContext(), DistinctAttr::create(builder.getUnitAttr()), compileUnitAttr, cuScope ? compileUnitAttr : scopeAttr, 
                    funcNameAttr, linkageNameAttr, 
                    file/*compileUnitAttr.getFile()*/, line, scopeLine, subprogramFlags, type);   

                debugScope.insert(SUBPROGRAM_DEBUG_SCOPE, subprogramAttr);
                debugScope.insert(DEBUG_SCOPE, subprogramAttr);

                return combine(functionLocation, subprogramAttr);
            }
        }

        return functionLocation;
    }

    void setLexicalBlock(mlir::Location blockLocation) {

        if (auto fileAttr = dyn_cast_or_null<mlir::LLVM::DIFileAttr>(debugScope.lookup(FILE_DEBUG_SCOPE)))
        {
            if (auto scopeAttr = dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(debugScope.lookup(DEBUG_SCOPE)))
            {
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
        }
    }    

    void setNamespace(mlir::Location namespaceLocation, StringRef namespaceName, bool exportSymbols) {
        if (auto scopeAttr = dyn_cast_or_null<mlir::LLVM::DIScopeAttr>(debugScope.lookup(DEBUG_SCOPE)))
        {        
            auto namespaceAttr = mlir::LLVM::DINamespaceAttr::get(
                builder.getContext(), builder.getStringAttr(namespaceName), scopeAttr, exportSymbols);

            debugScope.insert(NAMESPACE_DEBUG_SCOPE, namespaceAttr);
            debugScope.insert(DEBUG_SCOPE, namespaceAttr);
        }
    }

private:
    mlir::FusedLoc combine(mlir::Location location, mlir::LLVM::DIScopeAttr scope)
    {
        return mlir::FusedLoc::get(builder.getContext(), {location}, scope);          
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_DEBUGINFOHELPER_H_
