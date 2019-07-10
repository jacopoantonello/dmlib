Activate-Anaconda
if (-not $?) {
	exit
}
pip uninstall -y devwraps
if (-not $?) {
	exit
}
pip uninstall -y zernike
if (-not $?) {
	exit
}
pip uninstall -y devwraps
if (-not $?) {
	exit
}

git config core.autocrlf true
git config core.fileMode false

cd devwraps
rm dist\*.whl -ErrorAction SilentlyContinue
python setup.py bdist_wheel
if (-not $?) {
	exit
}
pip install (get-item .\dist\*.whl)
cd ..

cd zernike
rm dist\*.whl -ErrorAction SilentlyContinue
python setup.py bdist_wheel
if (-not $?) {
	exit
}
pip install (get-item .\dist\*.whl)
cd ..

rm dist\*.whl -ErrorAction SilentlyContinue
python setup.py bdist_wheel
if (-not $?) {
	exit
}
pip install (get-item .\dist\*.whl)

git config --unset core.autocrlf
git config --unset core.fileMode
