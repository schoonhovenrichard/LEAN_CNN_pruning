use super::edge::Edge;
use super::norm_group::NormGroup;

use std::cmp::min;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use pbr::ProgressBar;


type MinHeap<T> = BinaryHeap<Reverse<T>>;


pub fn is_forward_pointing(edges: &[Edge]) -> bool {
    for e in edges.iter() {
        if e.to <= e.from {
            return false;
        }
    }
    true
}

pub fn is_sorted(edges: &[Edge]) -> bool {
    for i in 1..edges.len() {
        let e0 = &edges[i - 1];
        let e1 = &edges[i];

        let e0 = (e0.from, e0.to);
        let e1 = (e1.from, e1.to);

        if e1 < e0 {
            return false;
        }
    }
    true
}

pub fn argmax(values: &[Option<f32>]) -> (usize, f32) {
    let mut largest_val = f32::NEG_INFINITY;
    let mut largest_idx: usize = 0;
    for (i, val) in values.iter().enumerate() {
        if let Some(val) = val {
            if &largest_val <= val {
                largest_val = *val;
                largest_idx = i;
            }
        }
    }
    (largest_idx, largest_val)
}

#[derive(Debug, Clone)]
pub struct FastGraph {
    norm_group: NormGroup,
    edges: Vec<Edge>,
    // Skippable edges that cannot be marked:
    skippable: Vec<bool>,
    marked: Vec<bool>,

    num_nodes: usize,
    num_marked: usize,
    node_all_incoming_edge_idxs: Vec<HashSet<usize>>,
    node_all_outgoing_edge_idxs: Vec<HashSet<usize>>,

    node_max_dist: Vec<Option<f32>>,
    node_max_incoming_edge_idx: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct Path {
    node_dist: Vec<Option<f32>>,
    node_incoming_edge_idx: Vec<Option<usize>>,
}

impl FastGraph {
    pub fn new(edges: Vec<Edge>, norm_group: NormGroup) -> FastGraph {
        assert!(!edges.is_empty());
        assert!(is_forward_pointing(&edges));
        assert!(is_sorted(&edges));

        let skippable = vec![false; edges.len()];
        let num_marked: usize = 0;
        let marked = vec![false; edges.len()];

        // For each node, prepare a hashset containing all its incoming edges;
        // Also, prepare a hashset containing all its outgoing edges.
        let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;
        let mut node_all_incoming_edge_idxs: Vec<HashSet<usize>> =
            vec![HashSet::with_capacity(edges.len() / n); n];
        let mut node_all_outgoing_edge_idxs: Vec<HashSet<usize>> =
            vec![HashSet::with_capacity(edges.len() / n); n];

        for (i, e) in edges.iter().enumerate() {
            if let Some(node_incoming_edge_set) = node_all_incoming_edge_idxs.get_mut(e.to) {
                node_incoming_edge_set.insert(i);
            }
            if let Some(node_outgoing_edge_set) = node_all_outgoing_edge_idxs.get_mut(e.from) {
                node_outgoing_edge_set.insert(i);
            }
        }

        let dist_tuple = Self::compute_node_distances(norm_group, &edges);
        let (node_max_dist, node_max_incoming_edge_idx) = dist_tuple;

        FastGraph {
            norm_group,
            edges,
            skippable,
            marked,
            num_nodes: n,
            num_marked,
            node_all_incoming_edge_idxs,
            node_all_outgoing_edge_idxs,
            node_max_dist,
            node_max_incoming_edge_idx,
        }
    }

    pub fn set_skippable(&mut self, skippable: &[bool]) {
        assert!(self.edges.len() == skippable.len());

        self.skippable = Vec::from(skippable);
    }


    fn compute_node_distances(
        norm_group: NormGroup,
        edges: &[Edge],
   ) -> (Vec<Option<f32>>, Vec<Option<usize>>) {
        let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;

        let mut node_dist: Vec<Option<f32>> = vec![None; n];
        let mut node_incoming_edge_idx: Vec<Option<usize>> = vec![None; n];

        // Compute node distances
        for (i, e) in edges.iter().enumerate() {
            let new_dist = norm_group.compute_dist(node_dist[e.from], &e);
            let old_dist = node_dist[e.to];

            if norm_group.do_replace(old_dist, new_dist) {
                node_dist[e.to] = Some(new_dist);
                node_incoming_edge_idx[e.to] = Some(i);
            }
        }
        (node_dist, node_incoming_edge_idx)
    }

    fn get_longest_path_edge_idxs(&self) -> Vec<usize> {
        let (largest_node_idx, _) = argmax(&self.node_max_dist);
        let mut cur_node_idx = largest_node_idx;

        // Walk backward from largest node to find all edges in longest path:
        let mut path_edge_idxs: Vec<usize> = Vec::with_capacity(self.num_nodes);

        while let Some(incoming_edge_idx) = self.node_max_incoming_edge_idx[cur_node_idx] {
            let incoming_edge = &self.edges[incoming_edge_idx];
            cur_node_idx = incoming_edge.from;
            path_edge_idxs.push(incoming_edge_idx);
        }
        path_edge_idxs
    }

    fn mark_edges(&mut self, edge_idxs: &[usize]) -> usize {
        let mut nodes_to_recalculate : MinHeap<usize> = MinHeap::with_capacity(self.num_nodes);

        let mut num_marked: usize = 0;

        // Mark edges, but note that the maximal distance of the destination
        // node of each edge has to be recalculated.
        for &edge_idx in edge_idxs.iter() {
            let is_marked = self.mark_single_edge(edge_idx);
            if is_marked {
                num_marked += 1;
            }
            let edge = &self.edges[edge_idx];
            nodes_to_recalculate.push(Reverse(edge.to));
        }

        // Recalculate downstream nodes
        while let Some(Reverse(cur_node_idx)) = nodes_to_recalculate.pop() {
            // We are uing a min-heap, so at any point in time, node_idx is the
            // smallest node that is in invalid state. Hence, after invalidating
            // the current computed distance, we can recalculate it on the basis
            // of the *valid* incoming edge and node values.
            self.node_max_dist[cur_node_idx] = None;
            self.node_max_incoming_edge_idx[cur_node_idx] = None;

            // Recalculate node_distance
            for &edge_idx in (&self.node_all_incoming_edge_idxs[cur_node_idx]).iter() {
                let edge = &self.edges[edge_idx];
                assert!(edge.to == cur_node_idx);

                let src_node_idx = edge.from;
                let current_dist = self.node_max_dist[cur_node_idx];
                let src_dist = self.node_max_dist[src_node_idx];
                let new_dist = self.norm_group.compute_dist(src_dist, &edge);

                if self.norm_group.do_replace(current_dist, new_dist) {
                    self.node_max_dist[cur_node_idx] = Some(new_dist);
                    self.node_max_incoming_edge_idx[cur_node_idx] = Some(edge_idx);
                }
            }

            // Mark downstream nodes for recalculation
            for &edge_idx in (&self.node_all_outgoing_edge_idxs[cur_node_idx]).iter() {
                // Recalculate downstream node if it is connected to current node.
                let downstream_node_idx = self.edges[edge_idx].to;
                if let Some(incoming_edge) = self.node_max_incoming_edge_idx[downstream_node_idx] {
                    if self.edges[incoming_edge].from == cur_node_idx {
                        nodes_to_recalculate.push(Reverse(downstream_node_idx));
                    }
                }
            }
        }

        num_marked
    }

    //// Mark a single edge and update graph connectivity
    //// NOTE: does *not* update node distances.
    //// Returns true if edge was marked, else false (if edge was skippable).
    fn mark_single_edge(&mut self, edge_idx: usize) -> bool {
        if self.skippable[edge_idx] {
            return false;
        }

        // Mark edge
        self.marked[edge_idx] = true;
        self.num_marked += 1;

        // Remove edge as incoming edge from destination node:
        let edge = &self.edges[edge_idx];
        let dest_node_idx = edge.to;
        // TODO: use `unwrap` or `expect`: this should never fail!
        if let Some(node_incoming_edge_set) = self.node_all_incoming_edge_idxs.get_mut(dest_node_idx)
        {
            node_incoming_edge_set.remove(&edge_idx);
        }

        // Remove edge as outgoing edge from source node:
        let src_node_idx = edge.from;
        // TODO: use `unwrap` or `expect`: this should never fail!
        if let Some(node_outgoing_edge_set) = self.node_all_outgoing_edge_idxs.get_mut(src_node_idx)
        {
            node_outgoing_edge_set.remove(&edge_idx);
        }
        true
    }
    // fn update_longest_path(&mut self) {
    //     let n = self.num_nodes;

    //     let mut node_dist: Vec<Option<f32>> = vec![None; n];
    //     let mut node_incoming_edge_idx: Vec<Option<usize>> = vec![None; n];

    //     // Compute node distances
    //     for (i, e) in self.edges.iter().enumerate() {
    //         let new_dist = self.compute_new_dist(node_dist[e.from], &e);
    //         let old_dist = node_dist[e.to];

    //         if self.do_replace(old_dist, new_dist) {
    //             node_dist[e.to] = Some(new_dist);
    //             node_incoming_edge_idx[e.to] = Some(i);
    //         }
    //     }

    //     self.longest_path = Some(Path{
    //         node_dist,
    //         node_incoming_edge_idx,

    //     })
    // }

    // fn iterate_back_longest_path<'a>(&'a self) -> Option<BackwardPathIterator<'a>> {
    //     // Compute argmax of node_dist
    //     if let Some(path) = &self.longest_path {
    //         let (largest_node_idx, _) = argmax(&path.node_dist);
    //         Some(
    //             BackwardPathIterator {
    //                 graph: &self,
    //                 path: &path,
    //                 node_idx: largest_node_idx,
    //             })
    //     } else {
    //         None
    //     }
    // }



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
    pub fn mark_longest_paths(mut self, num_to_mark: usize) -> Vec<bool> {
        let num_to_mark = min(num_to_mark, self.edges.len());

        // Create progress bar
        let mut pb = ProgressBar::new(num_to_mark as u64);

        while self.num_marked < num_to_mark {
            pb.set(self.num_marked as u64);

            // Get longest path
            let longest_path_edge_idxs = self.get_longest_path_edge_idxs();
            // Mark edges in longest path
            let path_num_marked = self.mark_edges(&longest_path_edge_idxs);

            // Break if we can't find paths anymore
            if path_num_marked == 0 {
                // All edges are 'negative' (either < 0 or < 1) and thus paths
                // cannot be formed anymore. Proceed to individual pruning..
                break;
            }
        }

        // Continue with individual pruning
        pb.message("Continuing with individual pruning!");
        let mut norm_idx_pairs: Vec<_> = self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, edge)| if self.marked[i] { None } else { Some((edge.len, i)) })
            .collect();

        // Sort from largest edge.len to smallest (hence `reverse`)
        norm_idx_pairs.sort_by(|(n1, _), (n2, _)| n1.partial_cmp(n2).unwrap().reverse());

        for (_, edge_idx) in norm_idx_pairs.iter() {
            pb.set(self.num_marked as u64);
            if num_to_mark <= self.num_marked {
                break;
            }
            self.mark_single_edge(*edge_idx);
        }
        pb.finish_print("done");

        self.marked
    }

}

pub fn fast_graph_mark_longest_paths(norm_group: NormGroup, edges: Vec<Edge>, num_to_mark: usize) -> Vec<bool> {
            FastGraph::new(edges.clone(), norm_group)
                .mark_longest_paths(num_to_mark)
}

mod tests {
    #[test]
    fn fast_graph_works() {
        use super::*;
        use crate::norm_group::NormGroup;

        let edges = vec![
            Edge::new(0, 1, 0.1),
            Edge::new(0, 1, 1.1),
            Edge::new(0, 1, 0.5),
            Edge::new(0, 2, 5.0),
            Edge::new(1, 2, 1.0),
        ];
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 1),
            vec![false, false, false, true, false]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 3),
            vec![false, true, false, true, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 4),
            vec![false, true, true, true, true]
        );
    }

    #[test]
    fn fast_graph_skippable_works() {
        use super::*;
        use crate::norm_group::NormGroup;

        let edges = vec![
            Edge::new(0, 1, 1.1),
            Edge::new(0, 1, 1.2),
            Edge::new(0, 1, 1.3),
        ];

        let skippable = vec![false, false, true];

        let mut graph = FastGraph::new(edges.clone(), NormGroup::Multiplicative);
        graph.set_skippable(&skippable);

        assert_eq!(
            graph.mark_longest_paths(1),
            vec![false, true, false]
        );
    }

    #[test]
    fn fast_graph_mult_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 1.0),
            Edge::new(0, 1, 2.0),
            Edge::new(0, 1, 3.0),
            Edge::new(0, 2, 7.0),
            Edge::new(1, 2, 3.0),
        ];
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 1),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 2),
            vec![false, false, true, false, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 3),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 4),
            vec![false, true, true, true, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 5),
            vec![true, true, true, true, true]
        );

        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 1),
            vec![false, false, false, true, false]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 2),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 3),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 4),
            vec![false, true, true, true, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 5),
            vec![true, true, true, true, true]
        );
    }

    #[test]
    fn longest_paths_faster_works() {
        use super::*;
        // Check that long chains are correctly handled:
        // Full chain is worth 5 * 5 / 2 = 12.5
        // sub-chains are worth as most 5.
        let edges = vec![
            Edge::new(0, 1, 5.0),
            Edge::new(1, 2, 0.5),
            Edge::new(2, 3, 5.0),
        ];
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 1),
            vec![true, true, true]
        );

        // Check that node distances are also forward propagated
        let edges = vec![
            Edge::new(0, 1, 2.0),
            Edge::new(1, 2, 0.5),
            Edge::new(2, 3, 1.5),
        ];
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 1),
            vec![true, false, false]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 2),
            vec![true, false, true]
        );
        assert_eq!(
            fast_graph_mark_longest_paths(NormGroup::Multiplicative, edges.clone(), 3),
            vec![true, true, true]
        );
    }

    #[test]
    #[ignore]
    fn longest_path_consistent() {
        use super::*;
        use crate::mark_longest_paths_faster;
        let n = 64;
        let mut edges: Vec<Edge> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(Edge::new(i, j, (i + j) as f32));
            }
        }

        let a = fast_graph_mark_longest_paths(NormGroup::Additive, edges.clone(), 20_000);
        let b = mark_longest_paths_faster(&edges, 20_000, NormGroup::Additive);
        assert_eq!(a, b);
    }
}
