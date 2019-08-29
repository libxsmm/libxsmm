@ECHO OFF
SETLOCAL

SET LIBXSMMROOT=%~d0%~p0..
SETX PATH "%LIBXSMMROOT%\lib\intel64;%LIBXSMMROOT%\lib\ia32;%PATH%"

IF EXIST C:\cygwin64\usr\include\eigen3\Eigen\Dense (
  SET EIGENROOT=C:\cygwin64\usr\include\eigen3
)
IF EXIST C:\blaze\blaze\Blaze.h (
  SET BLAZEROOT=C:\blaze
)

CALL "%~d0%~p0_vs.bat" 2019

ENDLOCAL
