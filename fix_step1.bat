@echo off
echo ================================================
echo  STEP 1: Aborting stuck merge state
echo ================================================
cd /d "f:\capstone project\predictive_maintenance\WILDESOUL-engine-maintenance-app"
git merge --abort 2>nul
git checkout -- . 2>nul
echo Merge state cleared.
echo.
echo Press any key to continue after merge is aborted...
pause
