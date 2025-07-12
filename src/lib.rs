pub mod algos;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub action: Actions,

    /// Precision of Pi to calculate in digits.
    #[arg(short, long, default_value_t = 1000)]
    pub prec: usize,

    /// Measure and show the runtime.
    #[arg(long)]
    pub measure_time: bool,

    /// Path of file to output and store the calculated Pi digits.
    #[arg(long)]
    pub output_to: Option<String>,
}

#[derive(Subcommand, Clone)]
pub enum Actions {
    /// Leibniz formula
    Leibniz,
    /// Bailey–Borwein–Plouffe formula
    BBP,
    /// Spigot Gosper algorithm
    SPG,
    /// Newton method. 9th order convergence.
    Newton,
    /// Borwein algorithm nonic (9th) convergence version
    BN,
    /// Brent–Salamin algorithm
    BS,
    /// Gauss–Legendre algorithm
    GL,
    /// Machin-like formulas (arctan)
    AG,
    /// Chudnovsky algorithm
    Chu,
    /// Chudnovsky algorithm with binary splitting
    CB,
    /// Chudnovsky algorithm with binary splitting and multi-thread
    CBP,
}

pub fn decide_iterations(act: Actions, decimal_prec: usize) -> u64 {
    match act {
        Actions::Leibniz => {
            if decimal_prec > 19 {
                panic!("Overflow! Please use a smaller precision (<= 19) for Leibniz formula.");
            }
            10_u64.pow(decimal_prec as u32) / 2
        }
        Actions::BBP => decimal_prec as u64,
        Actions::BN => {
            let log9 = 9_f64.log10();
            let p = decimal_prec as f64;
            let iters = p.log10() / log9;
            iters.ceil() as u64 + 1
        }
        Actions::BS => {
            let p = decimal_prec as f64;
            let iters = p.log2();
            iters.ceil() as u64 + 1
        }
        Actions::GL => {
            let p = decimal_prec as f64;
            let iters = p.log2();
            iters.ceil() as u64 + 1
        }
        Actions::Chu => decimal_prec as u64 / 14 + 1,
        Actions::CB => decimal_prec as u64 / 14 + 1,
        Actions::CBP => decimal_prec as u64 / 14 + 1,
        Actions::AG => 1,
        Actions::SPG => ((decimal_prec as f64 / 0.9).round() + 2.) as u64,
        Actions::Newton => {
            let log9 = 9_f64.log10();
            let p = decimal_prec as f64;
            let iters = p.log10() / log9;
            iters.ceil() as u64 + 1
        }
    }
}

pub fn get_iters_and_prec(act: Actions, decimal_prec: usize) -> (u64, u32) {
    let iters = decide_iterations(act, decimal_prec);
    let prec = (decimal_prec as f64 * 3.3219280948874).round() as u32 + 10;
    (iters, prec)
}

#[cfg(feature = "py")]
pub mod py_wrapper {
    use super::*;
    use pyo3::prelude::*;
    use rug::float::Round;

    /// Chudnovsky binary splitting algorithm with multi-threading.
    /// https://www.craig-wood.com/nick/articles/pi-chudnovsky/
    #[pyfunction]
    fn chudnovsky_binary_splitting(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::Chu, decimal_prec);
        let pi = algos::pi_chudnovsky_binary_splitting_parallel(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Newton method for π with 9th order convergence.
    /// https://www.hvks.com/Numerical/Downloads/HVE%20Practical%20implementation%20of%20PI%20Algorithms.pdf
    #[pyfunction]
    fn newton(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::Newton, decimal_prec);
        let pi = algos::pi_newton_9th(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Leibniz formula for π
    /// https://en.wikipedia.org/wiki/Leibniz_formula_for_π
    #[pyfunction]
    fn leibniz(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::Leibniz, decimal_prec);
        let pi = algos::pi_leibniz(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Bailey–Borwein–Plouffe formula for π
    /// https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula
    #[pyfunction]
    fn bailey_borwein_plouffe(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::BBP, decimal_prec);
        let pi = algos::pi_bailey_borwein_plouffe_v2(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Borwein's algorithm nonic convergence version for π
    /// https://en.wikipedia.org/wiki/Borwein%27s_algorithm#Nonic_convergence
    #[pyfunction]
    fn borwein_novin(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::BN, decimal_prec);
        let pi = algos::pi_borwein_novin(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Brent–Salamin algorithm for π
    /// https://mathworld.wolfram.com/Brent-SalaminFormula.html
    #[pyfunction]
    fn brent_salamin(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::BS, decimal_prec);
        let pi = algos::pi_brent_salamin(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Gauss–Legendre algorithm for π
    /// https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm
    #[pyfunction]
    fn gauss_legendre(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::GL, decimal_prec);
        let pi = algos::pi_gauss_legendre(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Machin-like formulas for π
    /// https://www.craig-wood.com/nick/articles/pi-machin/
    #[pyfunction]
    fn arctan_gauss(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::AG, decimal_prec);
        let pi = algos::pi_arctan_gauss(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Spigot algorithm for π with Gosper's seires
    /// https://www.gavalas.dev/blog/spigot-algorithms-for-pi-in-python/#using-gospers-series
    #[pyfunction]
    fn spigot_gosper(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::SPG, decimal_prec);
        let pi = algos::pi_spigot_gosper(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    /// Chudnovsky algorithm for π
    /// https://en.wikipedia.org/wiki/Chudnovsky_algorithm
    #[pyfunction]
    fn chudnovsky(decimal_prec: usize) -> PyResult<String> {
        let (iters, prec) = get_iters_and_prec(Actions::Chu, decimal_prec);
        let pi = algos::pi_chudnovsky(0, iters, prec);
        Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
    }

    #[pymodule]
    fn calc_pi(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(chudnovsky_binary_splitting, m)?)?;
        m.add_function(wrap_pyfunction!(newton, m)?)?;
        m.add_function(wrap_pyfunction!(leibniz, m)?)?;
        m.add_function(wrap_pyfunction!(bailey_borwein_plouffe, m)?)?;
        m.add_function(wrap_pyfunction!(borwein_novin, m)?)?;
        m.add_function(wrap_pyfunction!(brent_salamin, m)?)?;
        m.add_function(wrap_pyfunction!(gauss_legendre, m)?)?;
        m.add_function(wrap_pyfunction!(spigot_gosper, m)?)?;
        m.add_function(wrap_pyfunction!(arctan_gauss, m)?)?;
        m.add_function(wrap_pyfunction!(chudnovsky, m)?)?;
        Ok(())
    }
}
