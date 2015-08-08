@ECHO OFF
SETLOCAL

cd ..
bash -c "make realclean ; make M='1 2 3 4 5 23' header sources main"
cd ide

SET LIBXSMMROOT=%~d0%~p0\..

CALL %~d0"%~p0"_vs.bat vs2015
START %~d0"%~p0"_vs2015.sln

ENDLOCAL