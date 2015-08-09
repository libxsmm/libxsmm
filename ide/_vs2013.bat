@ECHO OFF
SETLOCAL

cd ..
bash -c "make realclean ; make M='1 2 3 4 5 23' header sources main"
cd ide

SET LIBXSMMROOT=%~d0%~p0\..

CALL %~d0"%~p0"_vs.bat vs2013
IF NOT "%VS_IDE%" == "" (
  START "" "%VS_IDE%" %~d0"%~p0"_vs2013.sln
) ELSE (
  START %~d0"%~p0"_vs2013.sln
)

ENDLOCAL