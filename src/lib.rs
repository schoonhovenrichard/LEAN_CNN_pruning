pub mod edge;
pub mod norm_group;
pub mod fast_graph;
pub mod python_bindings;

pub use edge::Edge;
pub use norm_group::NormGroup;
pub use fast_graph::{FastGraph, fast_graph_mark_longest_paths};

use NormGroup::*;
use fast_graph::{is_forward_pointing, is_sorted, argmax};
use pbr::ProgressBar;
use std::cmp::min;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

type MinHeap<T> = BinaryHeap<Reverse<T>>;



///////////////////////////////////////////////////////////////////////////////
//                                  Iterator                                 //
///////////////////////////////////////////////////////////////////////////////

// Implement an iterator to easily walk backwards over a directed path from a
// starting node. It is only used twice, so probably overkill to define an
// iterator for it...

struct BackwardEdgeIterator<'a> {
    edges: &'a [Edge],
    incoming_edge_idx: &'a [Option<usize>],
    node_idx: usize,
}

impl<'a> Iterator for BackwardEdgeIterator<'a> {
    type Item = (usize, &'a Edge);

    // The method that generates each item
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(incoming_edge_idx) = self.incoming_edge_idx[self.node_idx] {
            let incoming_edge = &self.edges[incoming_edge_idx];
            self.node_idx = incoming_edge.from;
            Some((incoming_edge_idx, incoming_edge))
        } else {
            None
        }
    }
}

fn iterate_back_from<'a>(
    edges: &'a [Edge],
    incoming_edge_idx: &'a [Option<usize>],
    node_idx: usize,
) -> BackwardEdgeIterator<'a> {
    BackwardEdgeIterator {
        edges,
        incoming_edge_idx,
        node_idx,
    }
}

///////////////////////////////////////////////////////////////////////////////
//                            Auxilliary functions                           //
///////////////////////////////////////////////////////////////////////////////

fn compute_node_distances(
    edges: &[Edge],
    norm_group: NormGroup,
) -> (Vec<Option<f32>>, Vec<Option<usize>>) {
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;

    let mut node_dist: Vec<Option<f32>> = vec![None; n];
    let mut node_incoming_edge_idx: Vec<Option<usize>> = vec![None; n];

    // Compute node distances
    for (i, e) in edges.iter().enumerate() {
        let new_dist = compute_dist(node_dist[e.from], &e, norm_group);
        let old_dist = node_dist[e.to];

        if do_replace(old_dist, new_dist, norm_group) {
            node_dist[e.to] = Some(new_dist);
            node_incoming_edge_idx[e.to] = Some(i);
        }
    }
    (node_dist, node_incoming_edge_idx)
}

fn compute_dist(src_node_dist: Option<f32>, edge: &Edge, norm_group: NormGroup) -> f32 {
    match norm_group {
        Additive => src_node_dist.unwrap_or(0.0) + edge.len,
        Multiplicative => src_node_dist.unwrap_or(1.0) * edge.len,
    }
}

/// Logic to determine whether an edge should be placed in a possible longest
/// path. If the current distance is `None`, then the edge is accepted if it has
/// positive weight (> 0.0 for additive, > 1.0 for multiplicative). Otherwise,
/// the edge is accepted if it has greater weight than the existing one.
fn do_replace(current_dist: Option<f32>, proposed_dist: f32, norm_group: NormGroup) -> bool {
    match current_dist {
        None => match norm_group {
            Additive => 0.0 <= proposed_dist,
            Multiplicative => 1.0 <= proposed_dist,
        },
        Some(current_dist) => current_dist <= proposed_dist,
    }
}

///////////////////////////////////////////////////////////////////////////////
//            Meat: mark edges that are part of some longest path            //
///////////////////////////////////////////////////////////////////////////////

/// Mark edges that are in a longest_path. Continue until at least `num_to_mark`
/// edges are marked. Once an edge has been marked, it is ignored for further
/// computations of longest paths.
pub fn mark_longest_paths(edges: &[Edge], num_to_mark: usize, norm_group: NormGroup) -> Vec<bool> {
    assert!(!edges.is_empty());
    assert!(is_forward_pointing(edges));
    assert!(is_sorted(edges));

    let num_to_mark = min(num_to_mark, edges.len());
    let mut num_marked: usize = 0;
    let mut marked = vec![false; edges.len()];

    // n - 1 is the index of the largest node
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;

    while num_marked < num_to_mark {
        let mut node_dist: Vec<Option<f32>> = vec![None; n];
        let mut node_incoming_edge_idx: Vec<Option<usize>> = vec![None; n];

        // Compute node distances
        for (i, e) in edges.iter().enumerate() {
            if marked[i] {
                continue;
            }

            let new_dist = norm_group.compute_dist(node_dist[e.from], e);
            let old_dist = node_dist[e.to];

            if norm_group.do_replace(old_dist, new_dist) {
                node_dist[e.to] = Some(new_dist);
                node_incoming_edge_idx[e.to] = Some(i);
            }
        }
        // Compute argmax of node_dist
        let (largest_node_idx, _longest_path_len) = argmax(&node_dist);

        // Walk backward from largest node to find all edges in longest path:
        let mut path_length = 0;
        for (edge_idx, _) in iterate_back_from(edges, &node_incoming_edge_idx, largest_node_idx) {
            marked[edge_idx] = true;
            path_length += 1;
        }
        num_marked += path_length;
    }

    marked
}


/// Mark edges that are in a longest_path. Continue until at least `num_to_mark`
/// edges are marked. Once an edge has been marked, it is ignored for further
/// computations of longest paths.
///
/// The `mark_longest_paths` function is quadratic in the number of edges. This
/// is a problem when it is applied to a network with 1-20 million edges. In
/// `mark_longest_paths_faster`, we make use of the realization that in our use
/// case the usual "longest path" consists of only 1-2 edges in the majority of
/// cases. It therefore makes sense to do a lot of bookkeeping to not have to
/// repeat the longest path algorithm from scratch.
pub fn mark_longest_paths_faster(
    edges: &[Edge],
    num_to_mark: usize,
    norm_group: NormGroup,
) -> Vec<bool> {
    assert!(!edges.is_empty());
    assert!(is_forward_pointing(edges));
    assert!(is_sorted(edges));

    let num_to_mark = min(num_to_mark, edges.len());
    let mut num_marked: usize = 0;
    let mut marked = vec![false; edges.len()];

    // For each node, prepare a hashset containing all its incoming edges;
    // Also, prepare a hashset containing all its outgoing edges.
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;
    let mut node_all_incoming_edge_idxs: Vec<HashSet<usize>> =
        vec![HashSet::with_capacity(edges.len() / n); n];
    let mut node_all_outgoing_edge_idxs: Vec<HashSet<usize>> =
        vec![HashSet::with_capacity(edges.len() / n); n];

    let mut pb = ProgressBar::new(edges.len() as u64);
    for (i, e) in edges.iter().enumerate() {
        if let Some(node_incoming_edge_set) = node_all_incoming_edge_idxs.get_mut(e.to) {
            node_incoming_edge_set.insert(i);
        }
        if let Some(node_outgoing_edge_set) = node_all_outgoing_edge_idxs.get_mut(e.from) {
            node_outgoing_edge_set.insert(i);
        }
        pb.tick();
    }
    pb.finish_print("Finished preparing.");

    let (node_dist, node_incoming_edge_idx) = compute_node_distances(edges, norm_group);
    let mut node_dist: Vec<Option<f32>> = node_dist;
    let mut node_incoming_edge_idx: Vec<Option<usize>> = node_incoming_edge_idx;

    // Create progress bar
    let mut pb = ProgressBar::new(num_to_mark as u64);

    while num_marked < num_to_mark {
        pb.set(num_marked as u64);
        // Compute argmax of node_dist
        let (largest_node_idx, _) = argmax(&node_dist);

        // Walk backward from largest node to find all edges in longest path:
        let mut path_marked_edges: Vec<usize> = Vec::with_capacity(n);
        for (edge_idx, _) in iterate_back_from(edges, &node_incoming_edge_idx, largest_node_idx) {
            path_marked_edges.push(edge_idx);
        }

        let num_marked_edges_in_path = path_marked_edges.len();
        num_marked += num_marked_edges_in_path;

        // Walk forward through path and
        // - mark edges along the path
        // - remove edges from hashsets
        // - mark nodes along path for invalidation
        let mut nodes_to_recalculate: MinHeap<usize> = MinHeap::with_capacity(n);
        while let Some(edge_idx) = path_marked_edges.pop() {
            // Mark edge
            marked[edge_idx] = true;
            let edge = &edges[edge_idx];

            // Remove edge as incoming edge from destination node:
            let dest_node_idx = edge.to;
            if let Some(node_incoming_edge_set) = node_all_incoming_edge_idxs.get_mut(dest_node_idx)
            {
                node_incoming_edge_set.remove(&edge_idx);
            }

            // Remove edge as outgoing edge from source node:
            let src_node_idx = edge.from;
            if let Some(node_outgoing_edge_set) = node_all_outgoing_edge_idxs.get_mut(src_node_idx)
            {
                node_outgoing_edge_set.remove(&edge_idx);
            }

            // Recalculate destination node's distance. This will be updated later.
            nodes_to_recalculate.push(Reverse(dest_node_idx));
        }

        // Recalculate downstream nodes
        while let Some(Reverse(dest_node_idx)) = nodes_to_recalculate.pop() {
            // We are uing a min-heap, so at any point in time, node_idx is the
            // smallest node that is in invalid state. Hence, after invalidating
            // the current computed distance, we can recalculate it on the basis
            // of the *valid* incoming edge and node values.
            node_dist[dest_node_idx] = None;
            node_incoming_edge_idx[dest_node_idx] = None;

            // Recalculate node_distance
            for &edge_idx in (&node_all_incoming_edge_idxs[dest_node_idx]).iter() {
                let edge = &edges[edge_idx];
                assert!(edge.to == dest_node_idx);
                let src_node_idx = edge.from;
                let current_dist = node_dist[dest_node_idx];
                let new_dist = compute_dist(node_dist[src_node_idx], &edge, norm_group);

                if do_replace(current_dist, new_dist, norm_group) {
                    node_dist[dest_node_idx] = Some(new_dist);
                    node_incoming_edge_idx[dest_node_idx] = Some(edge_idx);
                }
            }

            // Mark downstream nodes from recalculation
            for &edge_idx in (&node_all_outgoing_edge_idxs[dest_node_idx]).iter() {
                // Recalculate downstream node if it is connected to current node.
                let downstream_node_idx = edges[edge_idx].to;
                if let Some(incoming_edge) = node_incoming_edge_idx[downstream_node_idx] {
                    if edges[incoming_edge].from == dest_node_idx {
                        nodes_to_recalculate.push(Reverse(downstream_node_idx));
                    }
                }
            }
        }
        if num_marked_edges_in_path == 0 {
            // All edges are 'negative' (either < 0 or < 1) and thus paths
            // cannot be formed anymore. Proceed to individual pruning..
            break;
        }
    }

    // Continue with individual pruning
    pb.message("Continuing with individual pruning!");
    let mut norm_idx_pairs: Vec<_> = edges
        .iter()
        .enumerate()
        .filter_map(|(i, edge)| if marked[i] { None } else { Some((edge.len, i)) })
        .collect();

    // Sort from largest to smallest (hence `reverse`)
    norm_idx_pairs.sort_by(|(n1, _), (n2, _)| n1.partial_cmp(n2).unwrap().reverse());

    for (_, i) in norm_idx_pairs.iter() {
        pb.set(num_marked as u64);
        if num_to_mark <= num_marked {
            break;
        }
        marked[*i] = true;
        num_marked += 1;
    }
    pb.finish_print("done");
    marked
}

mod tests {
    #[test]
    fn longest_paths_additive_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 0.1),
            Edge::new(0, 1, 1.1),
            Edge::new(0, 1, 0.5),
            Edge::new(0, 2, 5.0),
            Edge::new(1, 2, 1.0),
        ];
        assert_eq!(
            mark_longest_paths(&edges, 1, Additive),
            vec![false, false, false, true, false]
        );
        assert_eq!(
            mark_longest_paths(&edges, 3, Additive),
            vec![false, true, false, true, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 4, Additive),
            vec![false, true, true, true, true]
        );
    }

    #[test]
    fn longest_paths_mult_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 1.0),
            Edge::new(0, 1, 2.0),
            Edge::new(0, 1, 3.0),
            Edge::new(0, 2, 7.0),
            Edge::new(1, 2, 3.0),
        ];
        assert_eq!(
            mark_longest_paths(&edges, 1, Multiplicative),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 2, Multiplicative),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 3, Multiplicative),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 4, Multiplicative),
            vec![false, true, true, true, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 5, Multiplicative),
            vec![true, true, true, true, true]
        );

        assert_eq!(
            mark_longest_paths(&edges, 1, Additive),
            vec![false, false, false, true, false]
        );
        assert_eq!(
            mark_longest_paths(&edges, 2, Additive),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 3, Additive),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 4, Additive),
            vec![false, true, true, true, true]
        );
        assert_eq!(
            mark_longest_paths(&edges, 5, Additive),
            vec![true, true, true, true, true]
        );
    }

    #[test]
    fn longest_paths_faster_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 1.0),
            Edge::new(0, 1, 2.0),
            Edge::new(0, 1, 3.0),
            Edge::new(0, 2, 7.0),
            Edge::new(1, 2, 3.0),
        ];
        assert_eq!(
            mark_longest_paths_faster(&edges, 1, Multiplicative),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 2, Multiplicative),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 3, Multiplicative),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 4, Multiplicative),
            vec![false, true, true, true, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 5, Multiplicative),
            vec![true, true, true, true, true]
        );

        assert_eq!(
            mark_longest_paths_faster(&edges, 1, Additive),
            vec![false, false, false, true, false]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 2, Additive),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 3, Additive),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 4, Additive),
            vec![false, true, true, true, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 5, Additive),
            vec![true, true, true, true, true]
        );

        // Check that long chains are correctly handled:
        // Full chain is worth 5 * 5 / 2 = 12.5
        // sub-chains are worth as most 5.
        let edges = vec![
            Edge::new(0, 1, 5.0),
            Edge::new(1, 2, 0.5),
            Edge::new(2, 3, 5.0),
        ];
        assert_eq!(
            mark_longest_paths_faster(&edges, 1, Multiplicative),
            vec![true, true, true]
        );

        // Check that node distances are also forward propagated
        let edges = vec![
            Edge::new(0, 1, 2.0),
            Edge::new(1, 2, 0.5),
            Edge::new(2, 3, 1.5),
        ];
        assert_eq!(
            mark_longest_paths_faster(&edges, 1, Multiplicative),
            vec![true, false, false]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 2, Multiplicative),
            vec![true, false, true]
        );
        assert_eq!(
            mark_longest_paths_faster(&edges, 3, Multiplicative),
            vec![true, true, true]
        );
    }

    #[test]
    #[ignore]                   // This test is slow..
    fn longest_path_consistent() {
        use super::*;
        let n = 64;
        let mut edges: Vec<Edge> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(Edge::new(i, j, (i + j) as f32));
            }
        }

        let a = mark_longest_paths(&edges, 20_000, Additive);
        let b = mark_longest_paths_faster(&edges, 20_000, Additive);
        assert_eq!(a, b);
    }
}
