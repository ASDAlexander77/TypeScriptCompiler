#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_

#include "TypeScript/LowerToLLVM/LocationHelper.h"
#include "TypeScript/LowerToLLVM/LLVMDebugInfo.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

// The DISubprogramAttr attached at MLIRGen time carries an empty
// DISubroutineTypeAttr: the DI types of the signature need LLVM layout
// information (LLVMTypeConverter) that only exists here, at lowering time.
// This helper rebuilds the subprogram with the real signature types and remaps
// every reference to the old scope inside the function - locations, block
// arguments, and debug-intrinsic attributes. Attributes that contain the old
// subprogram as a nested scope (lexical blocks, local variables, labels,
// nested subprograms) are rebuilt transitively by AttrTypeReplacer.
class LLVMDebugInfoHelperFixer
{
    mlir_ts::FuncOp funcOp;
    const LLVMTypeConverter *typeConverter;
    MLIRContext *context;
    CompileOptions& compileOptions;

public:
    LLVMDebugInfoHelperFixer(mlir_ts::FuncOp funcOp, const LLVMTypeConverter *typeConverter, CompileOptions& compileOptions)
        : funcOp(funcOp), typeConverter(typeConverter), compileOptions(compileOptions)
    {
        context = funcOp.getContext();
    }

    void fix() {
        auto location = funcOp->getLoc();
        auto funcLocWithSubprog = dyn_cast<mlir::FusedLocWith<mlir::LLVM::DISubprogramAttr>>(location);
        if (!funcLocWithSubprog)
        {
            return;
        }

        LocationHelper lh(context);
        LLVMTypeConverterHelper llvmtch(typeConverter);
        LLVMDebugInfoHelper di(context, llvmtch, compileOptions);

        auto oldMetadata = funcLocWithSubprog.getMetadata();

        auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(location);
        auto [line, column] = lineAndColumn;

        SmallVector <mlir::LLVM::DITypeAttr> resultTypes;
        for (auto resType : funcOp.getResultTypes())
        {
            auto diType = di.getDIType(location, {}, resType, file, line, file);
            resultTypes.push_back(diType);
        }

        if (funcOp.getArgumentTypes().size() > 0 &&  funcOp.getResultTypes().size() == 0)
        {
            // return type is null
            resultTypes.push_back(mlir::LLVM::DINullTypeAttr());
        }

        for (auto argType : funcOp.getArgumentTypes())
        {
            auto diType = di.getDIType(location, {}, argType, file, line, file);
            resultTypes.push_back(diType);
        }

        auto subroutineTypeAttr = mlir::LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, resultTypes);
        SmallVector<mlir::LLVM::DINodeAttr> retainedNodes; // TODO: review usage of it
        SmallVector<mlir::LLVM::DINodeAttr> annotations; // TODO: review usage of it
        // a fresh DistinctAttr id: references to the old subprogram outside this
        // function (if any) keep the old identity instead of colliding with the
        // rebuilt one at LLVM IR translation time
        auto subprogramAttr = mlir::LLVM::DISubprogramAttr::get(
            context,
            DistinctAttr::create(mlir::UnitAttr::get(context)),
            oldMetadata.getCompileUnit(),
            oldMetadata.getScope(),
            oldMetadata.getName(),
            oldMetadata.getLinkageName(),
            oldMetadata.getFile(),
            oldMetadata.getLine(),
            oldMetadata.getScopeLine(),
            oldMetadata.getSubprogramFlags(),
            subroutineTypeAttr,
            retainedNodes,
            annotations);

        LLVM_DEBUG(llvm::dbgs() << "\n!! new prog attr: " << subprogramAttr << "\n");

        mlir::AttrTypeReplacer replacer;
        replacer.addReplacement([&](mlir::LLVM::DISubprogramAttr attr) -> std::optional<mlir::Attribute> {
            if (attr == oldMetadata)
            {
                return subprogramAttr;
            }

            return std::nullopt;
        });

        replacer.recursivelyReplaceElementsIn(funcOp, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/false);
    }
};

}

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMDEBUGINFOFIXER_H_
