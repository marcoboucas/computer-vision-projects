lint:
	python -m pylint src notebooks
	python -m mypy src notebooks
	python -m flake8 src notebooks

test:
	pytest


install:
	pip install wheel
	pip install -r requirements.txt -f https://download.pytorch.org/whl/cu102/torch_stable.html
install-dev: install
	pip install -r requirements-dev.txt


clean:
	rm -r test.* test2.* build dist ./**/palabox.egg-info

clean-package:
	rm -r build dist ./**/palabox.egg-info &

package: clean-package
	python setup.py sdist bdist_wheel

ship-test:
	python -m twine upload --repository testpypi dist/*

ship:
	python -m twine upload dist/*

coverage:
	pytest --cov=palabox --cov-report=html tests/
	cd htmlcov && start "http://localhost:8000" && python -m http.server


check-gpu:
	python -m scripts.check_gpu

jupyter-theme:
	jt -t monokai -f fira -fs 13 -nf ptsans -nfs 11 -N -kl -cursw 5 -cursc r -cellw 95% -T