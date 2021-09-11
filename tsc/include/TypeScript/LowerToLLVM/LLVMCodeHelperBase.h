#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPERWRAP_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPERWRAP_H_

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

enum class MemoryAllocSet
{
    None,
    Zero
};

class LLVMCodeHelperBase
{
  protected:
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverter *typeConverter;

  public:
    LLVMCodeHelperBase(Operation *op, PatternRewriter &rewriter, TypeConverter *typeConverter)
        : op(op), rewriter(rewriter), typeConverter(typeConverter)
    {
    }

    template <typename T> void seekLast(Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                if (globalOp.valueAttr() && globalOp.valueAttr().isa<T>())
                {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            }
        };

        block->walk(lastUse);
    }

    void seekLast(Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                rewriter.setInsertionPointAfter(globalOp);
            }
        };

        block->walk(lastUse);
    }

    void seekLastWithBody(Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                if (globalOp.getInitializerBlock())
                {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            }
        };

        block->walk(lastUse);
    }

    std::string getStorageStringName(std::string value)
    {
        auto opHash = std::hash<std::string>{}(value);

        std::stringstream strVarName;
        strVarName << "s_" << opHash;

        return strVarName.str();
    }

  private:
    /// Return a value representing an access into a global string with the given
    /// name, creating the string if necessary.
    Value getOrCreateGlobalString_(StringRef name, StringRef value)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);

        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            seekLast<StringAttr>(parentModule.getBody());

            auto type = th.getArrayType(th.getI8Type(), value.size());
            global = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, rewriter.getStringAttr(value));
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getIndexType(), th.getIndexAttrValue(0));
        return rewriter.create<LLVM::GEPOp>(loc, th.getI8PtrType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
    }

  public:
    Value getOrCreateGlobalString(std::string value)
    {
        return getOrCreateGlobalString(getStorageStringName(value), value);
    }

    Value getOrCreateGlobalString(StringRef name, std::string value)
    {
        return getOrCreateGlobalString_(name, StringRef(value.data(), value.length() + 1));
    }

    LLVM::LLVMFuncOp getOrInsertFunction(const StringRef &name, const LLVM::LLVMFunctionType &llvmFnType)
    {
        auto parentModule = op->getParentOfType<ModuleOp>();

        if (auto funcOp = parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
        {
            return funcOp;
        }

        auto loc = op->getLoc();

        // Insert the printf function into the body of the parent module.
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());
        return rewriter.create<LLVM::LLVMFuncOp>(loc, name, llvmFnType);
    }

    template <typename T> Value _MemoryAlloc(mlir::Value sizeOfAlloc, MemoryAllocSet zero);
    Value MemoryAlloc(mlir::Value sizeOfAlloc, MemoryAllocSet zero = MemoryAllocSet::None)
    {
        return _MemoryAlloc<int>(sizeOfAlloc, zero);
    }

    Value MemoryAlloc(mlir::Type storageType, MemoryAllocSet zero = MemoryAllocSet::None)
    {
        TypeHelper th(rewriter);

        auto loc = op->getLoc();

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        return MemoryAlloc(sizeOfTypeValue, zero);
    }

    Value MemoryAllocBitcast(mlir::Type res, mlir::Type storageType, MemoryAllocSet zero = MemoryAllocSet::None)
    {
        auto loc = op->getLoc();

        auto alloc = MemoryAlloc(storageType, zero);
        auto val = rewriter.create<LLVM::BitcastOp>(loc, res, alloc);
        return val;
    }

    Value MemoryAllocBitcast(mlir::Type res, mlir::Value sizeOfAlloc, MemoryAllocSet zero = MemoryAllocSet::None)
    {
        auto loc = op->getLoc();

        auto alloc = MemoryAlloc(sizeOfAlloc, zero);
        auto val = rewriter.create<LLVM::BitcastOp>(loc, res, alloc);
        return val;
    }

    template <typename T> Value _MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc);
    Value MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc)
    {
        return _MemoryRealloc<int>(ptrValue, sizeOfAlloc);
    }

    Value MemoryReallocBitcast(mlir::Type res, mlir::Value ptrValue, mlir::Value sizeOfAlloc)
    {
        auto loc = op->getLoc();

        auto alloc = MemoryRealloc(ptrValue, sizeOfAlloc);
        auto val = rewriter.create<LLVM::BitcastOp>(loc, res, alloc);
        return val;
    }

    template <typename T> LogicalResult _MemoryFree(mlir::Value ptrValue);
    LogicalResult MemoryFree(mlir::Value ptrValue)
    {
        return _MemoryFree<int>(ptrValue);
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPERWRAP_H_
