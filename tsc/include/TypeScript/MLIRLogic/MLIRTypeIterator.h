#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEITERATOR_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEITERATOR_H_

#include "TypeScript/TypeScriptOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <functional>

namespace mlir_ts = mlir::typescript;

namespace typescript
{

class MLIRTypeIterator
{
  public:
    std::function<bool(mlir::Type)> f;

    MLIRTypeIterator(std::function<bool(mlir::Type)> f_) : f(f_)
    {
    }

    // TODO: add logic to resolve class type, interafce type, etc
    bool iterate(mlir::Type def)
    {
        if (!f(def))
            return false;

        auto result = llvm::TypeSwitch<mlir::Type, bool>(def)
                          .Case<mlir_ts::ArrayType>([&](mlir_ts::ArrayType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::BoundFunctionType>([&](mlir_ts::BoundFunctionType t) {
                              for (auto subType : t.getInputs())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              for (auto subType : t.getResults())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              return true;
                          })
                          .Case<mlir_ts::BoundRefType>([&](mlir_ts::BoundRefType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::ClassType>([&](mlir_ts::ClassType t) {
                              // TODO:
                              return true;
                          })
                          .Case<mlir_ts::ClassStorageType>([&](mlir_ts::ClassStorageType t) {
                              // TODO:
                              return true;
                          })
                          .Case<mlir_ts::ConstArrayType>([&](mlir_ts::ConstArrayType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::ConstArrayValueType>([&](mlir_ts::ConstArrayValueType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::ConstTupleType>([&](mlir_ts::ConstTupleType t) {
                              for (auto subType : t.getFields())
                              {
                                  if (!iterate(subType.type))
                                      return false;
                              }

                              return true;
                          })
                          .Case<mlir_ts::EnumType>([&](mlir_ts::EnumType t) {
                              if (!iterate(t.getElementType()))
                                  return false;

                              return true;
                          })
                          .Case<mlir_ts::FunctionType>([&](mlir_ts::FunctionType t) {
                              for (auto subType : t.getInputs())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              for (auto subType : t.getResults())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              return true;
                          })
                          .Case<mlir_ts::HybridFunctionType>([&](mlir_ts::HybridFunctionType t) {
                              for (auto subType : t.getInputs())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              for (auto subType : t.getResults())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              return true;
                          })
                          .Case<mlir_ts::InferType>([&](mlir_ts::InferType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::InterfaceType>([&](mlir_ts::InterfaceType t) {
                              // TODO:
                              return true;
                          })
                          .Case<mlir_ts::LiteralType>([&](mlir_ts::LiteralType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::OptionalType>([&](mlir_ts::OptionalType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::RefType>([&](mlir_ts::RefType t) {
                              if (!iterate(t.getElementType()))
                                  return false;
                              return true;
                          })
                          .Case<mlir_ts::TupleType>([&](mlir_ts::TupleType t) {
                              for (auto subType : t.getFields())
                              {
                                  if (!iterate(subType.type))
                                      return false;
                              }

                              return true;
                          })
                          .Case<mlir_ts::UnionType>([&](mlir_ts::UnionType t) {
                              for (auto subType : t.getTypes())
                              {
                                  if (!iterate(subType))
                                      return false;
                              }

                              return true;
                          })
                          .Case<mlir_ts::ValueRefType>([&](mlir_ts::ValueRefType t) {
                              if (!iterate(t.getElementType()))
                                  return false;

                              return true;
                          })
                          .Default([](mlir::Type) { return true; });

        return result;
    }
};

class MLIRTypeIteratorLogic
{
  public:
    MLIRTypeIteratorLogic() = default;

    void forEach(mlir::Type type, std::function<bool(mlir::Type)> f)
    {
        MLIRTypeIterator iter(f);
        iter.iterate(type);
    }

    bool some(mlir::Type type, std::function<bool(mlir::Type)> f)
    {
        auto result = false;
        forEach(type, [&](mlir::Type type) {
            result |= f(type);
            return !result;
        });

        return result;
    }

    bool every(mlir::Type type, std::function<bool(mlir::Type)> f)
    {
        auto result = true;
        forEach(type, [&](mlir::Type type) {
            result &= f(type);
            return result;
        });

        return result;
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRTYPEITERATOR_H_
