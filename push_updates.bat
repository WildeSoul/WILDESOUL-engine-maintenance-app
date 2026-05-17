@echo off
echo ================================================
<<<<<<< HEAD
echo  Pushing Advanced Predictive Maintenance to GitHub
echo ================================================
echo.
cd /d "f:\capstone project\predictive_maintenance\WILDESOUL-engine-maintenance-app"
echo [1/5] Removing mlruns from git tracking...
git rm -r --cached mlruns/ 2>nul
echo Done.
echo.
echo [2/5] Staging all changes...
git add -A
echo.
echo [3/5] Committing...
git commit -m "Advanced: Voting Ensemble, Plotly dashboard, multi-stage CI/CD, enhanced reports"
echo.
echo [4/5] Pulling remote...
git pull origin main --no-rebase 2>nul
echo.
echo [5/5] Force pushing...
git push origin main --force
echo.
echo ================================================
echo  DONE! Check: https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app/actions
=======
echo  FIXING: Removing large mlruns/ from Git history
echo ================================================
echo.

cd /d "f:\capstone project\predictive_maintenance\WILDESOUL-engine-maintenance-app"

echo [1/6] Removing mlruns/ from git tracking...
git rm -r --cached mlruns/ 2>nul
echo Done.

echo.
echo [2/6] Staging .gitignore and all changes...
git add -A

echo.
echo [3/6] Committing cleanup...
git commit -m "Fix: Remove mlruns/ from tracking, update .gitignore, upgrade UI and reports"

echo.
echo [4/6] Soft-resetting to squash merge commit...
git reset --soft HEAD~3

echo.
echo [5/6] Re-committing as single clean commit...
git commit -m "Upgrade: Premium Streamlit UI, enhanced reports, Dockerfile, README - mlruns excluded"

echo.
echo [6/6] Force pushing to GitHub...
git push origin main --force

echo.
echo ================================================
echo  SUCCESS! Check GitHub Actions at:
echo  https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app/actions
>>>>>>> 7cc97198d2d889b7206651cda6252f27f82fbb8f
echo ================================================
pause
