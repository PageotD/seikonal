//! Narrow band property queueandnode state tracking for FFM
//!
//! Based on:
//! - https://medium.com/coding-rust/max-heap-min-heap-priority-queue-with-custom-comparator-in-rust-2a6c5a6c1262
//! - https://stackoverflow.com/questions/39949939/how-can-i-implement-a-min-heap-of-f64-with-rusts-binaryheap

use std::cmp::Ordering;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;
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
        }); // equal time, tie-breaker on index
        // Smallest traveltime should pop first
        assert_eq!(heap.pop().unwrap().traveltime, 0.5);
        assert_eq!(heap.pop().unwrap().traveltime, 0.5);
        assert_eq!(heap.pop().unwrap().traveltime, 1.5);
        assert_eq!(heap.pop().unwrap().traveltime, 2.0);
        assert!(heap.pop().is_none());
    }
}
