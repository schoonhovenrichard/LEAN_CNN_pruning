/// Edge describes an edge from a node to another node with some weight or
/// length. The length may be negative.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub len: f32,
}

impl Edge {
    pub fn new(from: usize, to: usize, len: f32) -> Edge {
        Edge { from, to, len }
    }
}
