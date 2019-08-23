@ECHO OFF
SETLOCAL

ECHO ================================================================================
ECHO One-time configuration (Cygwin w/ GNU GCC, GNU Make, and Python needed in PATH)
ECHO When configured, it is sufficient to start _vs2019.bat or _vs2019.sln
cd ..
bash -c "make realclean ; make headers sources"
cd ide

CALL %~d0"%~p0"_vs2019.bat

ENDLOCAL