#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOCATIONHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOCATIONHELPER_H_

#include "llvm/Support/Path.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LocationHelper
{
  public:
    LocationHelper(MLIRContext *context) : context(context)
    {
    }

    std::pair<LLVM::DIFileAttr, std::tuple<size_t, size_t>> getLineAndColumnAndFile(mlir::FileLineColLoc location)
    {
        SmallString<256> FullName(location.getFilename());
        sys::path::remove_filename(FullName);
        auto file = mlir::LLVM::DIFileAttr::get(context, sys::path::filename(location.getFilename()), FullName);        

        return {file, {location.getLine(), location.getColumn()}};
    }

    std::pair<LLVM::DIFileAttr, std::tuple<size_t, size_t>> getLineAndColumnAndFile(mlir::Location location)
    {
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(location))
        {
            return getLineAndColumnAndFile(fusedLoc.getLocations().front());
        }
        else if (auto namedLoc = dyn_cast<mlir::NameLoc>(location))
        {
            return getLineAndColumnAndFile(namedLoc.getChildLoc());
        }
        else if (auto fileLineColLoc = dyn_cast<mlir::FileLineColLoc>(location))
        {
            return getLineAndColumnAndFile(fileLineColLoc);
        }
        else if (auto opaqueLoc = dyn_cast<mlir::OpaqueLoc>(location))
        {
            return getLineAndColumnAndFile(opaqueLoc.getFallbackLocation());
        }

        return {LLVM::DIFileAttr(), {0, 0}};
    }    

    static std::tuple<size_t, size_t> getLineAndColumn(mlir::FileLineColLoc location)
    {
        return {location.getLine(), location.getColumn()};
    }

    static std::tuple<size_t, size_t> getLineAndColumn(mlir::Location location)
    {
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(location))
        {
            return getLineAndColumn(fusedLoc.getLocations().front());
        }
        else if (auto namedLoc = dyn_cast<mlir::NameLoc>(location))
        {
            return getLineAndColumn(namedLoc.getChildLoc());
        }
        else if (auto fileLineColLoc = dyn_cast<mlir::FileLineColLoc>(location))
        {
            return getLineAndColumn(fileLineColLoc);
        }
        else if (auto opaqueLoc = dyn_cast<mlir::OpaqueLoc>(location))
        {
            return getLineAndColumn(opaqueLoc.getFallbackLocation());
        }

        return {0, 0};
    }  

  private:
    MLIRContext *context;    
};


class LLVMLocationHelper
{
  public:
    static std::pair<StringRef, std::tuple<size_t, size_t>> getLineAndColumnAndFileName(mlir::FileLineColLoc location)
    {
        return {location.getFilename(), {location.getLine(), location.getColumn()}};
    }

    static std::pair<StringRef, std::tuple<size_t, size_t>> getLineAndColumnAndFileName(mlir::Location location)
    {
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(location))
        {
            return getLineAndColumnAndFileName(fusedLoc.getLocations().front());
        }
        else if (auto namedLoc = dyn_cast<mlir::NameLoc>(location))
        {
            return getLineAndColumnAndFileName(namedLoc.getChildLoc());
        }
        else if (auto fileLineColLoc = dyn_cast<mlir::FileLineColLoc>(location))
        {
            return getLineAndColumnAndFileName(fileLineColLoc);
        }
        else if (auto opaqueLoc = dyn_cast<mlir::OpaqueLoc>(location))
        {
            return getLineAndColumnAndFileName(opaqueLoc.getFallbackLocation());
        }

        return {"", {0, 0}};
    }     
};

}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOCATIONHELPER_H_