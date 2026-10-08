//! Integration tests for `seikonal-core`.

use seikonal_core::SeikonalError;

#[test]
fn test_integration_error_handling() {
    let err = SeikonalError::InvalidDimensions { nx: 0, nz: 10 };
    assert!(err.to_string().contains("Invalid grid dimensions"));
}
