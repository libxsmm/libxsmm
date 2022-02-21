@ECHO OFF
SETLOCAL

ECHO ================================================================================
ECHO One-time configuration (Cygwin w/ GNU GCC, GNU Make, and Python needed in PATH)
ECHO When configured, it is sufficient to start _vs20xx.bat or _vs20xx.sln.
ECHO IMPORTANT: due to zero-config, configuration is not necessary anymore!
ECHO            One may terminate this configuration (CTRL-C)
ECHO            and simply start _vs20xx.bat or _vs20xx.sln.
PAUSE
cd ..
bash -c "make realclean ; make headers sources"
ECHO
ECHO Now start _vs20xx.bat or _vs20xx.sln!

ENDLOCAL