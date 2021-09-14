use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::exceptions::PyTypeError;
use crate::{FastGraph, Edge, NormGroup::*, mark_longest_paths_faster};

/// A Python module implemented in Rust.
#[pymodule]
fn n_longest_paths(_py: Python, m: &PyModule) -> PyResult<()> {
    // wrapper of `mark_longest_paths_faster`
    #[pyfn(m, "mark_longest_paths")]
    fn mark_longest_paths_py<'py>(
        py: Python<'py>,
        from: PyReadonlyArrayDyn<usize>,
        to: PyReadonlyArrayDyn<usize>,
        len: PyReadonlyArrayDyn<f32>,
        num_to_mark: usize,
    ) -> &'py PyArray1<bool> {
        let from = from.as_array();
        let to = to.as_array();
        let len = len.as_array();

        let mut edges: Vec<Edge> = Vec::with_capacity(from.len());

        for i in 0..from.len() {
            edges.push(Edge::new(from[i], to[i], len[i]));
        }

        let marked = mark_longest_paths_faster(&edges, num_to_mark, Multiplicative);

        marked.into_pyarray(py)
    }

    #[pyfn(m, "fast_graph_mark_longest_paths")]
    #[text_signature = "(src, dest, lengths, num_to_mark, length_type, skippable=None, /)"]
    fn fast_graph_mark_longest_paths_py<'py>(
        py: Python<'py>,
        src: PyReadonlyArrayDyn<usize>,
        dest: PyReadonlyArrayDyn<usize>,
        lengths: PyReadonlyArrayDyn<f32>,
        num_to_mark: usize,
        length_type : &str,
        skippable: Option<PyReadonlyArrayDyn<bool>>,
    ) -> PyResult<&'py PyArray1<bool>> {
        let from = src.as_array();
        let to = dest.as_array();
        let len = lengths.as_array();


        let mut edges: Vec<Edge> = Vec::with_capacity(from.len());

        for i in 0..from.len() {
            edges.push(Edge::new(from[i], to[i], len[i]));
        }

        let norm_group = match length_type {
            "multiplicative" => Ok(Multiplicative),
            "additive" => Ok(Additive),
            _ => Err(PyTypeError::new_err("Expected `length_type` to equal 'multiplicative' or 'additive'.")),
        }?;

        let mut graph = FastGraph::new(edges, norm_group);

        if let Some(skippable) = skippable {
            let skippable: Vec<bool> = skippable
                .as_array()
                .into_iter()
                .map(|&s| s)
                .collect();
            graph.set_skippable(&skippable);
        }

        let marked = graph.mark_longest_paths(num_to_mark);

        Ok(marked.into_pyarray(py))
    }

    Ok(())
}
