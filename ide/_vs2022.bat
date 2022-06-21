@ECHO OFF
SETLOCAL

SET "LIBXSMMROOT=%~d0%~p0.."
SET "PATH=%LIBXSMMROOT%\lib\intel64;%LIBXSMMROOT%\lib\ia32;%PATH%"

IF "%EIGENROOT%"=="" (
IF EXIST "C:\eigen\Eigen\Dense" (
  SET "EIGENROOT=C:\eigen"
))
IF "%EIGENROOT%"=="" (
IF EXIST "C:\cygwin64\usr\include\eigen3\Eigen\Dense" (
  SET "EIGENROOT=C:\cygwin64\usr\include\eigen3"
))
IF "%BLAZEROOT%"=="" (
IF EXIST "C:\blaze\blaze\Blaze.h" (
  SET "BLAZEROOT=C:\blaze"
))

CALL "%~d0%~p0_vs.bat" 2022

ENDLOCAL
