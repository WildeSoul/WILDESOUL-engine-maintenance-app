@echo off
echo ================================================
echo  Pushing to GitHub
echo ================================================
cd /d "f:\capstone project\predictive_maintenance\WILDESOUL-engine-maintenance-app"
echo [1/4] Removing mlruns from tracking...
git rm -r --cached mlruns/ 2>nul
echo [2/4] Staging...
git add -A
echo [3/4] Committing...
git commit -m "Advanced: Voting Ensemble, Plotly dashboard, multi-stage pipeline"
echo [4/4] Pushing...
git push origin main --force
echo ================================================
echo  DONE! Check GitHub Actions
echo ================================================
pause
