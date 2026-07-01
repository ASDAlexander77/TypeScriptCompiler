#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRPRINTER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRPRINTER_H_

#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/DOM.h"
#include "TypeScript/MLIRLogic/MLIRGenStore.h"
#include "TypeScript/MLIRLogic/MLIRTypeIterator.h"
#include "TypeScript/MLIRLogic/MLIRHelper.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/APSInt.h"

#include <functional>

#define DEBUG_TYPE "mlir"

namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRPrinter
{
  public:

    template <typename T, typename F>
    void printFuncType(T &out, F t)
    {
        out << "(";
        auto first = true;
        auto index = 0;
        auto size = t.getInputs().size();
        auto isVar = t.getIsVarArg();
        for (auto subType : t.getInputs())
        {
            if (!first)
            {
                out << ", ";
            }

            if (isVar && size == 1)
            {
                out << "...";
            }

            out << "p" << index << ": ";

            printType(out, subType);
            first = false;
            index ++;
            size --;
        }
        out << ") => ";

        if (t.getNumResults() == 0)
        {
            out << "void";
        }
        else if (t.getNumResults() == 1)
        {
            printType(out, t.getResults().front());
        }
        else
        {
            out << "[";
            auto first = true;
            for (auto subType : t.getResults())
            {
                if (!first)
                {
                    out << ", ";
                }

                printType(out, subType);
                first = false;
            }

            out << "]";
        }
    }

    template <typename T, typename TPL>
    void printFields(T &out, TPL t)
    {
        auto first = true;
        for (auto field : t.getFields())
        {
            if (!first)
            {
                out << ", ";
            }

            if (field.id)
            {
                printAttribute(out, field.id, true);
                out << ":";
            }

            printType(out, field.type);
            first = false;
        }
    }    

    template <typename T, typename TPL>
    void printTupleType(T &out, TPL t)
    {
        out << "[";
        printFields(out, t);
        out << "]";        
    }

    template <typename T, typename TPL>
    void printObjectType(T &out, TPL t)
    {
        out << "{";
        printFields(out, t);
        out << "}";        
    }

    template <typename T, typename U>
    void printUnionType(T &out, U t, const char *S)
    {
        auto first = true;
        for (auto subType : t.getTypes())
        {
            if (!first)
            {
                out << S;
            }

            printType(out, subType);
            first = false;
        }        
    }
    
    template <typename T>
    void printAttribute(T &out, mlir::Attribute attr, bool stringAsFieldName =  false)
    {
        llvm::TypeSwitch<mlir::Attribute>(attr)
            .template Case<mlir::StringAttr>([&](auto a) {
                if (stringAsFieldName)
                {
                    out << a.getValue().str().c_str();
                }
                else
                {
                    out << "\"";
                    out.write_escaped(a.str().c_str());
                    out << "\"";
                }
            })
            .template Case<mlir::FlatSymbolRefAttr>([&](auto a) {
                out << a.getValue().str().c_str();
            })            
            .template Case<mlir::IntegerAttr>([&](auto a) {
                SmallVector<char> Str;
                a.getValue().toStringUnsigned(Str);
                StringRef strRef(Str.data(), Str.size());
                out << strRef.str().c_str();
            })
            .template Case<mlir::FloatAttr>([&](auto a) {
                SmallVector<char> Str;
                a.getValue().toString(Str);
                StringRef strRef(Str.data(), Str.size());
                out << strRef.str().c_str();
            })            
            .Default([](mlir::Attribute a) { 
                LLVM_DEBUG(llvm::dbgs() << "\n!! Type print is not implemented for : " << a << "\n";);
                llvm_unreachable("not implemented");
            });
    }

    template <typename T>
    void printType(T &out, mlir::Type type)
    {
        llvm::TypeSwitch<mlir::Type>(type)
            .template Case<mlir_ts::ArrayType>([&](auto t) {
                printType(out, t.getElementType());
                out << "[]";
            })
            .template Case<mlir_ts::BoundFunctionType>([&](auto t) {
                printFuncType(out, t);
            })
            .template Case<mlir_ts::BoundRefType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::ClassType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::ClassStorageType>([&](auto t) {
                printTupleType(out, t);                                
            })
            .template Case<mlir_ts::InterfaceType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::ConstArrayType>([&](auto t) {
                printType(out, t.getElementType());
                out << "[]";
            })
            .template Case<mlir_ts::ConstArrayValueType>([&](auto t) {
                printType(out, t.getElementType());
                out << "[]";
            })
            .template Case<mlir_ts::ConstTupleType>([&](auto t) {
                printTupleType(out, t);
            })
            .template Case<mlir_ts::EnumType>([&](auto t) {
                //printType(out, t.getElementType());
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::FunctionType>([&](auto t) {
                printFuncType(out, t);
            })
            .template Case<mlir_ts::HybridFunctionType>([&](auto t) {
                printFuncType(out, t);
            })
            .template Case<mlir_ts::ConstructFunctionType>([&](auto t) {
                out << "new ";
                printFuncType(out, t);
            })
            .template Case<mlir_ts::ExtensionFunctionType>([&](auto t) {
                printFuncType(out, t);
            })            
            .template Case<mlir_ts::InferType>([&](auto t) {
                out << "infer ";
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::LiteralType>([&](auto t) {
                printAttribute(out, t.getValue());
                // printType(out, t.getElementType());
            })
            .template Case<mlir_ts::OptionalType>([&](auto t) {
                printType(out, t.getElementType());
                out << " | undefined";
            })
            .template Case<mlir_ts::RefType>([&](auto t) {
                out << "Reference<";
                printType(out, t.getElementType());
                out << ">";
            })
            .template Case<mlir_ts::TupleType>([&](auto t) {
                printTupleType(out, t);
            })
            .template Case<mlir_ts::UnionType>([&](auto t) {
                printUnionType(out, t, " | ");
            })
            .template Case<mlir_ts::IntersectionType>([&](auto t) {
                printUnionType(out, t, " & ");
            })
            .template Case<mlir_ts::ValueRefType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::ConditionalType>([&](auto t) {
                printType(out, t.getCheckType());
                out << "extends";
                printType(out, t.getCheckType());
                out << " ? ";
                printType(out, t.getTrueType());
                out << " : ";
                printType(out, t.getFalseType());
            })
            .template Case<mlir_ts::IndexAccessType>([&](auto t) {
                printType(out, t.getType());
                out << "[";
                printType(out, t.getIndexType());
                out << "]";
            })
            .template Case<mlir_ts::KeyOfType>([&](auto t) {
                out << "keyof ";
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::MappedType>([&](auto t) {
                out << "[";
                printType(out, t.getElementType());
                out << " of ";
                printType(out, t.getNameType());
                out << " extends ";
                printType(out, t.getConstrainType());
                out << "]";
            })
            .template Case<mlir_ts::TypeReferenceType>([&](auto t) {
                printAttribute(out, t.getName());
                if (t.getTypes().size() > 0)
                {
                    out << "<";
                    auto first = true;
                    for (auto subType : t.getTypes())
                    {
                        if (!first)
                        {
                            out << ", ";
                        }

                        printType(out, subType);
                        first = false;
                    }
                    out << ">";
                }
            })
            .template Case<mlir_ts::TypePredicateType>([&](auto t) {
                printType(out, t.getElementType());
            })
            .template Case<mlir_ts::NamedGenericType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir_ts::ObjectType>([&](auto t) {
                out << "object";
            })
            .template Case<mlir_ts::ObjectStorageType>([&](auto t) {
                printTupleType(out, t);       
            })
            .template Case<mlir_ts::NeverType>([&](auto) { 
                out << "never";
            })
            .template Case<mlir_ts::UnknownType>([&](auto) {
                out << "unknown";
            })
            .template Case<mlir_ts::AnyType>([&](auto) {
                out << "any";
            })
            .template Case<mlir_ts::NumberType>([&](auto) {
                out << "number";
            })
            .template Case<mlir_ts::StringType>([&](auto) {
                out << "string";
            })
            .template Case<mlir_ts::BooleanType>([&](auto) {
                out << "boolean";
            })
            .template Case<mlir_ts::UndefinedType>([&](auto t) {
                out << "undefined";
            })
            .template Case<mlir_ts::VoidType>([&](auto) {
                out << "void";
            })
            .template Case<mlir_ts::ByteType>([&](auto) {
                out << "byte";
            })
            .template Case<mlir_ts::CharType>([&](auto) {
                out << "char";
            })
            .template Case<mlir_ts::OpaqueType>([&](auto) {
                out << "Opaque";
            })
            .template Case<mlir_ts::ConstType>([&](auto) {
                out << "const";
            })
            .template Case<mlir_ts::SymbolType>([&](auto) {
                out << "Symbol";
            })             
            .template Case<mlir_ts::NullType>([&](auto) {
                out << "null";
            })
            .template Case<mlir_ts::BigIntType>([&](auto) {
                out << "bigint";
            })
            .template Case<mlir_ts::NamespaceType>([&](auto t) {
                out << t.getName().getValue().str().c_str();
            })
            .template Case<mlir::NoneType>([&](auto) {
                out << "void";
            })
            .template Case<mlir::IntegerType>([&](auto t) {
                if (t.isSigned())
                    out << "s";
                else if (t.isSignless())
                    out << "i";
                else
                    out << "u";
                out << t.getIntOrFloatBitWidth();
            })
            .template Case<mlir::FloatType>([&](auto t) {
                out << "f" << t.getIntOrFloatBitWidth();
            })
            .template Case<mlir::IndexType>([&](auto) {
                out << "index";
            })
            .Default([&](mlir::Type t) { 
                LLVM_DEBUG(llvm::dbgs() << "\n!! Type print is not implemented for : " << t << "\n";);
                out << t;
            });
    }
};

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRPRINTER_H_
