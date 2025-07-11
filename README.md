# Pi Calculation in Rust (*Also available in Python*)

[![Crates.io](https://img.shields.io/crates/v/calc_pi.svg)](https://crates.io/crates/calc_pi)
[![PyPI](https://img.shields.io/pypi/v/calc-pi.svg?color=blue)](https://pypi.org/project/calc-pi/)

This is a Rust CLI tool for calculating Pi digits using various algorithms. It can compute to a million digits of Pi in less than 1 second (see [below](#performance)).

Supported algorithms are listed as follows:

<!-- | Algorithm | Note | -->
| Algorithm | Note |
| --------- | --- |
| [Leibniz](https://en.wikipedia.org/wiki/Leibniz_formula_for_π) | Slowest |
| [Bailey-Borwein-Plouffe (BBP)](https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula) | |
| [Spigot Gosper](https://www.gavalas.dev/blog/spigot-algorithms-for-pi-in-python/#using-gospers-series) | |
| [Newton 9th Order Convergence](https://www.hvks.com/Numerical/Downloads/HVE%20Practical%20implementation%20of%20PI%20Algorithms.pdf) | Page 9 of the paper |
| [Borwein's Formula](https://en.wikipedia.org/wiki/Borwein%27s_algorithm#Nonic_convergence) | |
| [Brent-Salamin](https://mathworld.wolfram.com/Brent-SalaminFormula.html) | |
| [Gauss-Legendre](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm) | Only slightly different from Brent-Salamin. |
| [Machin-like Formula](https://en.wikipedia.org/wiki/Machin-like_formula#Two-term_formulas) | Using arctan to calculate Pi. |
| [Chudnovsky](https://en.wikipedia.org/wiki/Chudnovsky_algorithm) | |
| [Chudnovsky Binary Splitting](https://www.craig-wood.com/nick/articles/pi-chudnovsky/) | World record-breaking algorithm. |
| [Chudnovsky Binary Splitting Parallelized](https://yamakuramun.info/2024/05/26/686/) | Fastest. |

## Installation

**Use in terminal**
```bash
cargo install calc_pi
```

**Use in Python**
```bash
pip install calc-pi
```
```python
import calc_pi

# Calculate Pi to 1000 digits using Chudnovsky algorithm
print(calc_pi.chudnovsky(1000))  # 3.1415926535897932...
```

## Usage

```bash
CLI for calculating millions of Pi digits within seconds. Various algorithms are supported.

Usage: calc_pi [OPTIONS] <COMMAND>

Commands:
  leibniz  Leibniz formula
  bbp      Bailey–Borwein–Plouffe formula
  spg      Spigot Gosper algorithm
  newton   Newton method. 9th order convergence
  bn       Borwein algorithm nonic (9th) convergence version
  bs       Brent–Salamin algorithm
  gl       Gauss–Legendre algorithm
  ag       Machin-like formulas (arctan)
  chu      Chudnovsky algorithm
  cb       Chudnovsky algorithm with binary splitting
  cbp      Chudnovsky algorithm with binary splitting and multi-thread
  help     Print this message or the help of the given subcommand(s)

Options:
  -p, --prec <PREC>            Precision of Pi to calculate in digits [default: 1000]
      --measure-time           Measure and show the runtime
      --output-to <OUTPUT_TO>  Path of file to output and store the calculated Pi digits
  -h, --help                   Print help
  -V, --version                Print version
```

```bash
# Calculate Pi with Chudnovsky binary splitting algorithm to 1,000,000 digits.
calc_pi -p 1000000 cbp

# Show the runtime.
calc_pi -p 1000000 --measure-time cbp

# Output the result to file
calc_pi -p 1000000 --output-to pi.txt cbp
```

## Recommended Reading
- [Practical implementation of PI Algorithms](https://www.hvks.com/Numerical/Downloads/HVE%20Practical%20implementation%20of%20PI%20Algorithms.pdf): This paper provides a comprehensive overview of various algorithms for calculating Pi, including their implementation details and performance comparisons.
- [Craig Wood's article on Chudnovsky algorithm](https://www.craig-wood.com/nick/articles/pi-chudnovsky): A detailed explanation of the Chudnovsky algorithm, including its mathematical background and implementation.

## Performance

**Test Environment**
- OS: Ubuntu 20.04
- CPU: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
- RAM: 16GB
- Rust Version: 1.86.0
- Cargo Version: 1.86.0

| Algorithm | 9 digits |  1K digits | 10K digits | 100K digits | 1M digits | 10M digits |
| --------- | -------- |  --------- | ---------- | ----------- | --------- | ---------  |
| Leibniz   | 28.568 s | -          | -          | -           | -         | -          |
| Bailey-Borwein-Plouffe | - | 2.5 ms | 660.3 ms | 208.9 s     | -         | -          |
| Spigot Gosper | -    | 3.1 ms     | 867.6 ms   | 250.217 s   | -         | -          |
| Newton    | -        | 5.2 ms     | 43.7 ms    | 1.421 s     | 30.475 s  | -          |
| Borwein Nonic | -    | 3.6 ms     | 159.6 ms   | 5.123 s     | 68.424 s  | -          |
| Brent-Salamin | -    | -          | 3.2 ms     | 70.8 ms     | 1.231 s   | 21.364 s   |
| Gauss-Legendre | -   | -          | 3.5 ms     | 70.6 ms     | 1.229 s   | 21.121 s   |
| Machin-like Formula  | - | 1.9 ms | 14.9 ms    | 465.4 ms    | 8.673 s   | -          |
| Chudnovsky | -       | -          | 5.1 ms     | 369.2 ms    | 38.22 s   | -          |
| Chudnovsky Binary Splitting | - | - | 2.0 ms   | 23.5 ms     | 434.1 ms  | 7.9 s      |
| Chudnovsky Binary Splitting Parallelized | - | - | - | 19.9 ms | 293.7 ms | 4.628 s   |

| Algorithm | 100M digits |
| --------- | ----------- |
| Chudnovsky Binary Splitting | 109.58 s |
| Chudnovsky Binary Splitting Parallelized | 48.739 s |
| [y-cruncher](https://www.numberworld.org/y-cruncher/) (single-thread) | 21.908 s |
| [y-cruncher](https://www.numberworld.org/y-cruncher/) (multi-thread) | 5.354 s |

## Notes
- Parallel version of Chudnovsky Binary Splitting uses half of the CPU cores available on the system.
- Use `Integer` for variables when possible. This boosts performance significantly.

## Reference
- https://www.hvks.com/Numerical/Downloads/HVE%20Practical%20implementation%20of%20PI%20Algorithms.pdf
- https://www.craig-wood.com/nick/articles/pi-chudnovsky
- https://github.com/Pencilcaseman/iron_pi
- https://yamakuramun.info/2024/05/26/686/
- https://qiita.com/yonaka15/items/992b3306106c150f36c6
- https://github.com/elkasztano/piday25
- https://www.numberworld.org/y-cruncher/
