macro(set_Options)

if(MSVC)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else()
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch -Wno-unused-function -Wno-sign-compare -Wno-implicit-fallthrough -lstdc++fs")
endif()

endmacro()


