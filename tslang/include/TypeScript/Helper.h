#ifndef TYPESCRIPT_HELPER_H
#define TYPESCRIPT_HELPER_H

#include <memory>
#include "llvm/Support/Casting.h" // Add this header

template< typename T, typename U >
inline std::unique_ptr< T > dynamic_pointer_cast(std::unique_ptr< U > &&ptr) {
    U * const stored_ptr = ptr.release();
    
    // Check type using LLVM's custom, RTTI-free system
    if (stored_ptr && llvm::isa< T >(stored_ptr)) {
        return std::unique_ptr< T >(static_cast< T * >(stored_ptr));
    }
    else {
        if (stored_ptr) {
            throw "Invalid cast";
        }
        ptr.reset(stored_ptr);
        return std::unique_ptr< T >();
    }
}

#endif // TYPESCRIPT_HELPER_H
