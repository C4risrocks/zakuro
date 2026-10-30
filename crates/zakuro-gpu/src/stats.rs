//! what drew, for the kpi: draw calls by whose vertex shader ran on the
//! host GPU against ones the CPU shaded first, and the frames shown. the
//! runtime logs a session kpi line with these when it says good-bye, which
//! the log file keeps for the docs' kpi table.

use std::sync::atomic::{AtomicU64, Ordering};

static SHADED_GPU: AtomicU64 = AtomicU64::new(0);
static SHADED_CPU: AtomicU64 = AtomicU64::new(0);

/// a draw call whose vertex shader ran on the host GPU.
pub fn shaded_gpu() {
    SHADED_GPU.fetch_add(1, Ordering::Relaxed);
}

/// a draw call the CPU shaded before raster went at it: the software
/// renderer's draws as well, and those of a GPU that does not shade.
pub fn shaded_cpu() {
    SHADED_CPU.fetch_add(1, Ordering::Relaxed);
}

/// one cpu shading counted above hands over to the re-walk, which counts
/// it again: the draw with flushed textures counts once, not twice.
pub fn shaded_cpu_hands_over() {
    SHADED_CPU.fetch_sub(1, Ordering::Relaxed);
}

/// (draws the GPU shaded, draws the CPU shaded, per cent of them on the
/// GPU, none when no draw reached a rasterizer).
pub fn geometry() -> (u64, u64, Option<u64>) {
    let gpu = SHADED_GPU.load(Ordering::Relaxed);
    let cpu = SHADED_CPU.load(Ordering::Relaxed);
    let total = gpu + cpu;
    (gpu, cpu, (total > 0).then(|| gpu * 100 / total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_cent_of_shaded_draws() {
        // the tiler's own tests run alongside, so the counts are not
        // fresh: the test measures its own steps' deltas
        let (gpu_was, cpu_was, _) = geometry();
        shaded_gpu();
        let (gpu, cpu, _) = geometry();
        assert_eq!((gpu, cpu), (gpu_was + 1, cpu_was));
        shaded_cpu();
        let (gpu, cpu, per_cent) = geometry();
        assert_eq!((gpu, cpu), (gpu_was + 1, cpu_was + 1));
        assert_eq!(per_cent, Some(gpu * 100 / (gpu + cpu)));
    }
}
