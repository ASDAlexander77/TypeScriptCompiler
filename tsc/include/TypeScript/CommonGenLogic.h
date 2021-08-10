#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_H_

#include "file_helper.h"
#include "parser.h"

#ifndef NDEBUG
#define DBG_DUMP(o) LLVM_DEBUG(o->dump(););
#define DBG_PRINT_BLOCK(block)                                                                                                             \
    LLVM_DEBUG(llvm::dbgs() << "\n === block dump === \n");                                                                                \
    LLVM_DEBUG(block->dump(););
#define DBG_PRINT                                                                                                                          \
    LLVM_DEBUG(llvm::dbgs() << "\n === START OF DEBUG === \n");                                                                            \
    LLVM_DEBUG(llvm::dbgs() << "\n*** region: " << rewriter.getInsertionBlock()->getParent() << "\n");                                     \
    for (auto &block : *rewriter.getInsertionBlock()->getParent())                                                                         \
    {                                                                                                                                      \
        LLVM_DEBUG(llvm::dbgs() << "\n === block dump === \n");                                                                            \
        LLVM_DEBUG(block.dump(););                                                                                                         \
    }                                                                                                                                      \
    LLVM_DEBUG(llvm::dbgs() << "\n === END OF DEBUG === \n");
#define DBG_TYPE(l, t) LLVM_DEBUG(llvm::dbgs() << l##" type: " << t << "\n");
#else
#define DBG_PRINT_BLOCK(block)
#define DBG_PRINT
#endif

namespace mlir_ts = mlir::typescript;

namespace typescript
{
class MLIRHelper
{
  public:
    static std::string getName(ts::Identifier identifier)
    {
        std::string nameValue;
        if (identifier)
        {
            nameValue = wstos(identifier->escapedText);
        }

        return nameValue;
    }

    static std::string getName(ts::PrivateIdentifier identifier)
    {
        std::string nameValue;
        if (identifier)
        {
            nameValue = wstos(identifier->escapedText);
        }

        return nameValue;
    }

    static std::string getName(ts::Node name)
    {
        std::string nameValue;
        if (name == SyntaxKind::Identifier)
        {
            return getName(name.as<ts::Identifier>());
        }

        if (name == SyntaxKind::PrivateIdentifier)
        {
            return getName(name.as<ts::PrivateIdentifier>());
        }

        return nameValue;
    }

    static mlir::StringRef getName(ts::Node name, llvm::BumpPtrAllocator &stringAllocator)
    {
        auto nameValue = getName(name);
        return mlir::StringRef(nameValue).copy(stringAllocator);
    }

    static std::string getAnonymousName(mlir::Location loc)
    {
        // auto calculate name
        std::stringstream ssName;
        ssName << "__uf" << hash_value(loc);
        return ssName.str();
    }

    static bool matchLabelOrNotSet(mlir::StringAttr loopLabel, mlir::StringAttr opLabel)
    {
        auto loopHasValue = loopLabel && loopLabel.getValue().size() > 0;
        auto opLabelHasValue = opLabel && opLabel.getValue().size() > 0;

        if (!opLabelHasValue)
        {
            return true;
        }

        if (loopHasValue && opLabelHasValue)
        {
            auto eq = loopLabel.getValue() == opLabel.getValue();
            return eq;
        }

        return false;
    }

    static bool matchSimilarTypes(mlir::Type ty1, mlir::Type ty2)
    {
        if (ty1 == ty2)
        {
            return true;
        }

        if (auto constArray1 = ty1.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            if (auto constArray2 = ty2.dyn_cast_or_null<mlir_ts::ConstArrayType>())
            {
                return matchSimilarTypes(constArray1.getElementType(), constArray2.getElementType());
            }
        }

        return false;
    }
};

enum class MatchResultType
{
    Match,
    NotMatchArgCount,
    NotMatchArg,
    NotMatchResultCount,
    NotMatchResult
};

struct MatchResult
{
    MatchResultType result;
    unsigned index;
};

class MLIRTypeHelper
{
    mlir::MLIRContext *context;

  public:
    MLIRTypeHelper(mlir::MLIRContext *context) : context(context)
    {
    }

    mlir::Type getI32Type()
    {
        return mlir::IntegerType::get(context, 32);
    }

    mlir::IntegerAttr getStructIndexAttrValue(int32_t value)
    {
        return mlir::IntegerAttr::get(getI32Type(), mlir::APInt(32, value));
    }

    bool isValueType(mlir::Type type)
    {
        return type && (type.isIntOrIndexOrFloat() || type.isa<mlir_ts::TupleType>());
    }

    mlir::Type convertConstTypeToType(mlir::Type type, bool &copyRequired)
    {
        if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            copyRequired = true;
            return mlir_ts::ArrayType::get(constArrayType.getElementType());
        }

        /*
        // tuple is value and copied already
        if (auto constTupleType = type.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            copyRequired = true;
            return mlir_ts::TupleType::get(context, constTupleType.getFields());
        }
        */

        copyRequired = false;
        return type;
    }

    MatchResult TestFunctionTypesMatch(mlir::FunctionType inFuncType, mlir::FunctionType resFuncType, unsigned startParam = 0)
    {
        // TODO: make 1 common function
        if (inFuncType.getInputs().size() != resFuncType.getInputs().size())
        {
            return {MatchResultType::NotMatchArgCount, 0};
        }

        for (unsigned i = startParam, e = inFuncType.getInputs().size(); i != e; ++i)
        {
            if (inFuncType.getInput(i) != resFuncType.getInput(i))
            {
                return {MatchResultType::NotMatchArg, i};
            }
        }

        auto inRetCount = inFuncType.getResults().size();
        auto resRetCount = resFuncType.getResults().size();

        auto noneType = mlir::NoneType::get(context);
        auto voidType = mlir_ts::VoidType::get(context);

        for (auto retType : inFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                inRetCount--;
            }
        }

        for (auto retType : resFuncType.getResults())
        {
            auto isVoid = !retType || retType == noneType || retType == voidType;
            if (isVoid)
            {
                resRetCount--;
            }
        }

        if (inRetCount != resRetCount)
        {
            return {MatchResultType::NotMatchResultCount, 0};
        }

        for (unsigned i = 0, e = inRetCount; i != e; ++i)
        {
            auto inRetType = inFuncType.getResult(i);
            auto resRetType = resFuncType.getResult(i);

            auto isInVoid = !inRetType || inRetType == noneType || inRetType == voidType;
            auto isResVoid = !resRetType || resRetType == noneType || resRetType == voidType;
            if (!isInVoid && !isResVoid && inRetType != resRetType)
            {
                return {MatchResultType::NotMatchResult, i};
            }
        }

        return {MatchResultType::Match, 0};
    }

    mlir::Value GetReferenceOfLoadOp(mlir::Value value)
    {
        if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(value.getDefiningOp()))
        {
            // this LoadOp will be removed later as unused
            auto refValue = loadOp.reference();
            return refValue;
        }

        return mlir::Value();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_H_
