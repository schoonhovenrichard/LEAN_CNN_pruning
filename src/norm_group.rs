use super::edge::Edge;

/// An awkwardly named enum to describe whether edge weights should be added or
/// multiplied when multiple edges are combined...
#[derive(Debug, Clone, Copy)]
pub enum NormGroup {
    Additive,
    Multiplicative,
}

use NormGroup::*;

impl NormGroup {
    pub fn compute_dist(&self, src_node_dist: Option<f32>, edge: &Edge) -> f32 {
        match self {
            Additive => src_node_dist.unwrap_or(0.0) + edge.len,
            Multiplicative => src_node_dist.unwrap_or(1.0) * edge.len,
        }
    }

    /// Logic to determine whether an edge should be placed in a possible longest
    /// path. If the current distance is `None`, then the edge is accepted if it has
    /// positive weight (> 0.0 for additive, > 1.0 for multiplicative). Otherwise,
    /// the edge is accepted if it has greater weight than the existing one.
    pub fn do_replace(&self, current_dist: Option<f32>, proposed_dist: f32) -> bool {
        match current_dist {
            None => match self {
                Additive => 0.0 <= proposed_dist,
                Multiplicative => 1.0 <= proposed_dist,
            },
            Some(current_dist) => current_dist <= proposed_dist,
        }
    }


}
