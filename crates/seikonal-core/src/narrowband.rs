//! Narrow band property queueandnode state tracking for FFM
//!
//! Based on:
//! - https://medium.com/coding-rust/max-heap-min-heap-priority-queue-with-custom-comparator-in-rust-2a6c5a6c1262
//! - https://stackoverflow.com/questions/39949939/how-can-i-implement-a-min-heap-of-f64-with-rusts-binaryheap

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Grid node state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeState {
    /// Node is Far: not reached by the wavefront (T=inf)
    #[default]
    Far,
    /// Node is in narrow band: on the wavefront
    NarrowBand,
    /// Node is fixed: travaltime is finalized and frozen
    Alive,
}

/// Implements reverse ordering to obtain min-heap
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedNode {
    /// Linear index of the node in the grid
    pub index: usize,
    /// Candidate traveltime at the node
    pub traveltime: f64,
}

impl Eq for OrderedNode {}

impl Ord for OrderedNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (small traveltimes first)
        other
            .traveltime
            .total_cmp(&self.traveltime)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for OrderedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Narrow band structure to manage the priority queue and node states.
#[derive(Debug, Clone)]
pub struct NarrowBand {
    heap: BinaryHeap<OrderedNode>,
    states: Vec<NodeState>,
}
impl NarrowBand {
    /// Creates a new `NarrowBand` tracker for a grid of `total_nodes`.
    ///
    /// All nodes are initially in the `Far` state.
    pub fn new(total_nodes: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            states: vec![NodeState::Far; total_nodes],
        }
    }

    /// Returns the states of a node by its linear index.
    #[inline]
    pub fn state(&self, index: usize) -> NodeState {
        self.states[index]
    }

    /// Sets the state of a node by its linear index.
    #[inline]
    pub fn set_state(&mut self, index: usize, state: NodeState) {
        self.states[index] = state;
    }

    /// Pushes a node into the narrow band priority queue with candidate `traveltime`.
    ///
    /// And update the node state to `NodeState::NarrowBand`.
    #[inline]
    pub fn push(&mut self, index: usize, traveltime: f64) {
        self.states[index] = NodeState::NarrowBand;
        self.heap.push(OrderedNode { index, traveltime });
    }

    /// Extracts the next minimum traveltime node from the narrow band.
    ///
    /// Returns `Some((index, traveltime))` if the narrow band is non-empty.
    /// Uses lazy deletion: stale entries (`Alive`) are automatically skipped.
    pub fn pop_min(&mut self, current_traveltimes: &[f64]) -> Option<(usize, f64)> {
        while let Some(ordered_node) = self.heap.pop() {
            let idx = ordered_node.index;
            // If the node is already fixed (Alive), skip
            if self.states[idx] == NodeState::Alive {
                continue;
            }
            // If entry is outdated compared to the currently recorded best time, skip
            if ordered_node.traveltime > current_traveltimes[idx] {
                continue;
            }
            return Some((idx, ordered_node.traveltime));
        }
        None
    }

    /// Returns true if there are no nodes currently pending in the heap.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the total number of entries in the underlying heap.
    #[inline]
    pub fn heap_len(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_node_state_default() {
        assert_eq!(NodeState::default(), NodeState::Far);
    }
    #[test]
    fn test_ordered_node_min_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(OrderedNode {
            index: 1,
            traveltime: 1.5,
        });
        heap.push(OrderedNode {
            index: 2,
            traveltime: 0.5,
        });
        heap.push(OrderedNode {
            index: 3,
            traveltime: 2.0,
        });
        heap.push(OrderedNode {
            index: 4,
            traveltime: 0.5,
        });
        assert_eq!(heap.pop().unwrap().traveltime, 0.5);
        assert_eq!(heap.pop().unwrap().traveltime, 0.5);
        assert_eq!(heap.pop().unwrap().traveltime, 1.5);
        assert_eq!(heap.pop().unwrap().traveltime, 2.0);
        assert!(heap.pop().is_none());
    }
    #[test]
    fn test_narrowband_operations_and_lazy_deletion() {
        let total_nodes = 5;
        let mut nb = NarrowBand::new(total_nodes);
        let mut traveltimes = vec![f64::INFINITY; total_nodes];
        assert_eq!(nb.state(0), NodeState::Far);
        // Push node 0 with T=1.0 and node 1 with T=2.0
        traveltimes[0] = 1.0;
        nb.push(0, 1.0);
        traveltimes[1] = 2.0;
        nb.push(1, 2.0);
        assert_eq!(nb.state(0), NodeState::NarrowBand);
        assert_eq!(nb.state(1), NodeState::NarrowBand);
        // Update node 1 with a better time T=0.5 (pushed again to heap)
        traveltimes[1] = 0.5;
        nb.push(1, 0.5);
        // First pop should yield node 1 with T=0.5
        let (idx, t) = nb.pop_min(&traveltimes).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(t, 0.5);
        nb.set_state(1, NodeState::Alive);
        // Second pop should yield node 0 with T=1.0
        let (idx, t) = nb.pop_min(&traveltimes).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(t, 1.0);
        nb.set_state(0, NodeState::Alive);
        // The stale entry for node 1 with T=2.0 must be skipped automatically
        assert!(nb.pop_min(&traveltimes).is_none());
    }
}
