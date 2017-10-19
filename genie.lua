solution "jefc"
    configurations { "Debug", "Release" }
    platforms      { "x32", "x64" }
    includedirs    { "include" }
    flags          { "NoPCH", "NoRTTI" , "ExtraWarnings" } 
    location       ( _ACTION )

    configuration { "Release" }
        defines { "NDEBUG" }

    project "jefc-lib"
        language "C"
        kind "StaticLib"
        files { "include/**.h",
                "lib/**.c", 
                "lib/**.h" }
        
        configuration { "vs* or windows" }
            defines "_CRT_SECURE_NO_WARNINGS"

        configuration "Debug*"
            flags { "Symbols" }
            
        configuration "Release*"
            flags { "Optimize" }
            
        configuration { "x64", "Debug" }
            targetdir ( path.join( "bin", "64", "debug" ) )
        
        configuration { "x64", "Release" }
            targetdir ( path.join( "bin", "64", "release" ) )
         
        configuration { "x32", "Debug" }
            targetdir ( path.join( "bin", "32", "debug" ) )

        configuration { "x32", "Release" }
            targetdir ( path.join( "bin", "32", "release" ) )

        configuration {}
