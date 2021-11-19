lint:
	python -m pylint palabox
	python -m mypy palabox
	python -m flake8 palabox

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