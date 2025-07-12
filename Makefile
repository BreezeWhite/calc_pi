build:
	@cargo build -r

publish:
	@cargo publish

py-install:
	@maturin develop --features py

py-publish:
	@maturin publish --features py

py-test:
	@python -c "import calc_pi; print(calc_pi.newton(100))"