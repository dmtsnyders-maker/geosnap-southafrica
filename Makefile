.PHONY: test check wheel clean

test:
	python -m pytest -q

check:
	python -m compileall -q geosnap_southafrica

wheel:
	python -m pip wheel . --no-deps --no-build-isolation -w dist

clean:
	rm -rf build dist *.egg-info .pytest_cache geosnap_southafrica/__pycache__ tests/__pycache__
