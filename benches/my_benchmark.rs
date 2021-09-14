use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use n_longest_paths::*;

pub fn bm_longest_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("longest_paths");

    for n in [100, 200, 300, 400, 500, 600, 700].iter() {
        for &step in [1, 2, 4, 8].iter() {
            let mut edges: Vec<Edge> = Vec::with_capacity(n * n);
            for i in 0..*n {
                for j in ((i + 1)..*n).step_by(step) {
                    edges.push(Edge::new(i, j, (i + j) as f32));
                }
            }
            // Skip expensive benchmarks
            if edges.len() > 40_000 {
                continue;
            }

            group.bench_with_input(
                BenchmarkId::new(
                    format!("mark_longest_paths step skip {}", step),
                    edges.len(),
                ),
                &n,
                |b, &_n| {
                    b.iter(|| mark_longest_paths(&edges, edges.len() * 10 / 9, NormGroup::Additive))
                },
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!("mark_longest_paths_faster step skip {}", step),
                    edges.len(),
                ),
                &n,
                |b, &_n| {
                    b.iter(|| {
                        mark_longest_paths_faster(&edges, edges.len() * 10 / 9, NormGroup::Additive)
                    })
                },
            );
        }
    }
}

criterion_group!(benches, bm_longest_paths);
criterion_main!(benches);
