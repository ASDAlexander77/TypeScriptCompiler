include_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc-8.0.4/include/")

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
if(MSVC)
  link_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/${CMAKE_BUILD_TYPE}/")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
elseif(MSVC)
  link_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc/${CMAKE_BUILD_TYPE_LOWERCASE}/")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()

add_mlir_library(TypeScriptAsyncRuntime
  STATIC
  AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR
)
