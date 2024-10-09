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

    std::pair<LLVM::DIFileAttr, size_t> getLineAndFile(mlir::FileLineColLoc location)
    {
        SmallString<256> FullName(location.getFilename());
        sys::path::remove_filename(FullName);
        auto file = mlir::LLVM::DIFileAttr::get(context, sys::path::filename(location.getFilename()), FullName);        

        return std::make_pair(file, location.getLine());
    }

    std::pair<LLVM::DIFileAttr, size_t> getLineAndFile(mlir::FusedLoc location)
    {
        auto locs = location.getLocations();
        if (locs.size() > 0)
        {
            if (auto fileLineColLoc = locs[0].dyn_cast<mlir::FileLineColLoc>())
            {
                return getLineAndFile(fileLineColLoc);
            }
        }
            
        return std::make_pair(LLVM::DIFileAttr(), 0);
    }

    std::pair<LLVM::DIFileAttr, size_t> getLineAndFile(mlir::Location location)
    {
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(location))
        {
            return getLineAndFile(fusedLoc);
        }
            
        return std::make_pair(LLVM::DIFileAttr(), 0);
    }    

    static size_t getLine(mlir::FileLineColLoc location)
    {
        return location.getLine();
    }

    static size_t getLine(mlir::FusedLoc location)
    {
        auto line = 0;

        auto locs = location.getLocations();
        if (locs.size() > 0)
        {
            if (auto fileLineColLoc = locs[0].dyn_cast<mlir::FileLineColLoc>())
            {
                line = getLine(fileLineColLoc);
            }
        }
            
        return line;
    }

    static size_t getLine(mlir::Location location)
    {
        auto line = 0;

        mlir::TypeSwitch<mlir::LocationAttr>(location)
            .Case<mlir::FusedLoc>([&](auto locParam) {
                line = getLine(locParam);
            }
        );       
            
        return line;
    }

    static std::tuple<size_t, size_t> getLineAndColumn(mlir::FileLineColLoc location)
    {
        return {location.getLine(), location.getColumn()};
    }

    static std::tuple<size_t, size_t> getLineAndColumn(mlir::FusedLoc location)
    {
        std::tuple<size_t, size_t> lineAndColumn = {0, 0};

        auto locs = location.getLocations();
        if (locs.size() > 0)
        {
            if (auto fileLineColLoc = locs[0].dyn_cast<mlir::FileLineColLoc>())
            {
                lineAndColumn = getLineAndColumn(fileLineColLoc);
            }
        }
            
        return lineAndColumn;
    }

    static std::tuple<size_t, size_t> getLineAndColumn(mlir::Location location)
    {
        std::tuple<size_t, size_t> lineAndColumn = {0, 0};

        mlir::TypeSwitch<mlir::LocationAttr>(location)
            .Case<mlir::FusedLoc>([&](auto locParam) {
                lineAndColumn = getLineAndColumn(locParam);
            }
        );       
            
        return lineAndColumn;
    }

  private:
    MLIRContext *context;    
};

class LLVMLocationHelper
{
  public:
    LLVMLocationHelper()
    {
    }

    std::pair<StringRef, size_t> getLineAndFile(mlir::FileLineColLoc location)
    {
        SmallString<256> FullName(location.getFilename());
        sys::path::remove_filename(FullName);
        auto file = sys::path::filename(location.getFilename());        

        return std::make_pair(file, location.getLine());
    }

    std::pair<StringRef, size_t> getLineAndFile(mlir::FusedLoc location)
    {
        auto locs = location.getLocations();
        if (locs.size() > 0)
        {
            if (auto fileLineColLoc = locs[0].dyn_cast<mlir::FileLineColLoc>())
            {
                return getLineAndFile(fileLineColLoc);
            }
        }
            
        return std::make_pair(StringRef(""), 0);
    }

    std::pair<StringRef, size_t> getLineAndFile(mlir::Location location)
    {
        if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(location))
        {
            return getLineAndFile(fusedLoc);
        }
            
        return std::make_pair(StringRef(""), 0);
    }    

    static size_t getLine(mlir::FileLineColLoc location)
    {
        return location.getLine();
    }

    static size_t getLine(mlir::FusedLoc location)
    {
        auto line = 0;

        auto locs = location.getLocations();
        if (locs.size() > 0)
        {
            if (auto fileLineColLoc = locs[0].dyn_cast<mlir::FileLineColLoc>())
            {
                line = getLine(fileLineColLoc);
            }
        }
            
        return line;
    }

    static size_t getLine(mlir::Location location)
    {
        auto line = 0;

        mlir::TypeSwitch<mlir::LocationAttr>(location)
            .Case<mlir::FusedLoc>([&](auto locParam) {
                line = getLine(locParam);
            }
        );       
            
        return line;
    }
};

}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LOCATIONHELPER_H_