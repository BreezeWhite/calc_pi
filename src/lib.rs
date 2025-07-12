pub mod algos;

use clap::{Parser, Subcommand};
use pyo3::prelude::*;
use rug::float::Round;

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

#[pyfunction]
fn chudnovsky_binary_splitting(decimal_prec: usize) -> PyResult<String> {
    let iters = decide_iterations(Actions::Chu, decimal_prec);
    let prec = (decimal_prec as f64 * 3.3219280948874).round() as u32 + 10;
    let pi = algos::pi_chudnovsky_binary_splitting_parallel(0, iters, prec);
    Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
}

#[pyfunction]
fn newton(decimal_prec: usize) -> PyResult<String> {
    let iters = decide_iterations(Actions::Newton, decimal_prec);
    let prec = (decimal_prec as f64 * 3.3219280948874).round() as u32 + 10;
    let pi = algos::pi_newton_9th(0, iters, prec);
    Ok(pi.to_string_radix_round(10, Some(decimal_prec), Round::Down))
}

#[pymodule]
fn calc_pi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chudnovsky_binary_splitting, m)?)?;
    m.add_function(wrap_pyfunction!(newton, m)?)?;
    Ok(())
}
