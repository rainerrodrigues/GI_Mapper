use backend::models::{DataPoint, DetectClustersRequest};
use serde_json::json;

#[tokio::test]
async fn test_cluster_detection_request_serialization() {
    let request = DetectClustersRequest {
        data_points: vec![
            DataPoint {
                id: "point1".to_string(),
                latitude: 28.6139,
                longitude: 77.2090,
                value: 100.0,
            },
            DataPoint {
                id: "point2".to_string(),
                latitude: 28.7041,
                longitude: 77.1025,
                value: 150.0,
            },
        ],
        algorithm: "auto".to_string(),
        parameters: None,
    };

    let json_str = serde_json::to_string(&request).unwrap();
    assert!(json_str.contains("point1"));
    assert!(json_str.contains("auto"));
}

#[tokio::test]
async fn test_data_point_creation() {
    let point = DataPoint {
        id: "test_point".to_string(),
        latitude: 28.6139,
        longitude: 77.2090,
        value: 100.0,
    };

    assert_eq!(point.id, "test_point");
    assert_eq!(point.latitude, 28.6139);
    assert_eq!(point.longitude, 77.2090);
    assert_eq!(point.value, 100.0);
}

#[tokio::test]
async fn test_cluster_request_with_parameters() {
    let request_json = json!({
        "data_points": [
            {
                "id": "p1",
                "latitude": 28.6,
                "longitude": 77.2,
                "value": 50.0
            }
        ],
        "algorithm": "dbscan",
        "parameters": {
            "epsilon": 0.5,
            "min_points": 5
        }
    });

    let request: DetectClustersRequest = serde_json::from_value(request_json).unwrap();
    assert_eq!(request.algorithm, "dbscan");
    assert!(request.parameters.is_some());
}

#[tokio::test]
#[ignore] // Requires running database
async fn test_cluster_database_operations() {
    use backend::db::Database;
    use serde_json::json;

    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set for tests");

    let db = Database::new(&database_url).await.unwrap();

    // Test creating a cluster
    let boundary = vec![
        vec![77.0, 28.0],
        vec![78.0, 28.0],
        vec![78.0, 29.0],
        vec![77.0, 29.0],
        vec![77.0, 28.0],
    ];
    let centroid = vec![77.5, 28.5];
    let parameters = json!({"epsilon": 0.5, "min_points": 5});
    let quality_metrics = json!({
        "silhouette_score": 0.65,
        "davies_bouldin_index": 0.85,
        "calinski_harabasz_score": 150.0
    });

    let cluster_id = backend::db::cluster::create_cluster(
        db.pool(),
        1,
        &boundary,
        &centroid,
        "DBSCAN",
        &parameters,
        10,
        0.5,
        1000.0,
        &quality_metrics,
        Some("test_hash"),
        Some("test_tx_id"),
    )
    .await
    .unwrap();

    // Test retrieving the cluster
    let cluster = backend::db::cluster::get_cluster_by_id(db.pool(), cluster_id)
        .await
        .unwrap();

    assert_eq!(cluster.cluster_id, 1);
    assert_eq!(cluster.algorithm, "DBSCAN");
    assert_eq!(cluster.member_count, 10);
    assert_eq!(cluster.centroid, vec![77.5, 28.5]);

    // Test listing clusters
    let clusters = backend::db::cluster::list_clusters(db.pool(), 10, 0)
        .await
        .unwrap();

    assert!(!clusters.is_empty());

    // Test counting clusters
    let count = backend::db::cluster::count_clusters(db.pool())
        .await
        .unwrap();

    assert!(count > 0);
}
