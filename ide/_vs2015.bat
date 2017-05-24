@ECHO OFF
SETLOCAL

cd ..
bash -c "make realclean ; make headers sources"
cd ide

SET LIBXSMMROOT=%~d0%~p0..

IF EXIST C:\cygwin64\usr\include\eigen3\Eigen\Dense (
  SET EIGENROOT=C:\cygwin64\usr\include\eigen3
)

CALL %~d0"%~p0"_vs.bat vs2015
IF NOT "%VS_IDE%" == "" (
  START "" "%VS_IDE%" "%~d0%~p0_vs2015.sln"
) ELSE (
  START %~d0"%~p0"_vs2015.sln
)

ENDLOCAL