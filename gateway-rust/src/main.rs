use axum::{
    extract::{DefaultBodyLimit, Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose, Engine as _};
use redis::AsyncCommands;
use serde::Serialize;
use std::sync::Arc;
use tower_http::limit::RequestBodyLimitLayer;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    redis_client: redis::Client,
}

#[derive(Serialize)]
struct TaskResponse {
    task_id: String,
    status: String,
}

#[derive(Serialize)]
struct StatusResponse {
    task_id: String,
    status: String,
    result: Option<serde_json::Value>,
    error: Option<String>,
}

#[tokio::main]
async fn main() {
    // 1. Initialize Tracing & Structured Logs
    tracing_subscriber::fmt::init();

    // 2. Resolve Port (Default to 7860 for Hugging Face Spaces compatibility)
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(7860);

    // 3. Resolve Redis Connection URL (Supports full REDIS_URL or host/port)
    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| {
        let host = std::env::var("REDIS_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("REDIS_PORT").unwrap_or_else(|_| "6379".to_string());
        format!("redis://{}:{}", host, port)
    });

    let redis_client = redis::Client::open(redis_url.clone())
        .unwrap_or_else(|e| panic!("Failed to create Redis client for {}: {}", redis_url, e));

    let state = Arc::new(AppState { redis_client });

    // 4. Configure Application Router
    let app = Router::new()
        .route("/process", post(submit_task))
        .route("/status/:task_id", get(get_status))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB limit
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!("🚀 Rust Axum Ingestion Gateway running on http://{}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn submit_task(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut conn = state
        .redis_client
        .get_async_connection()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let task_id = Uuid::new_v4().to_string();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        if field.name() == Some("file") {
            let filename = field.file_name().unwrap_or("document.pdf").to_string();
            let extension = filename.split('.').last().unwrap_or("pdf").to_string();
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

            let base64_data = general_purpose::STANDARD.encode(&data);
            let task_key = format!("task:{}", task_id);

            // Atomic task registration
            let _: () = redis::cmd("HSET")
                .arg(&task_key)
                .arg("status")
                .arg("queued")
                .arg("filename")
                .arg(&filename)
                .arg("extension")
                .arg(&extension)
                .arg("data")
                .arg(&base64_data)
                .query_async(&mut conn)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            // Push to FIFO task queue
            let _: () = conn
                .lpush("ocr_tasks", &task_id)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            return Ok((
                StatusCode::ACCEPTED,
                Json(TaskResponse {
                    task_id,
                    status: "queued".to_string(),
                }),
            ));
        }
    }

    Err((
        StatusCode::BAD_REQUEST,
        "Missing required multipart field 'file'".to_string(),
    ))
}

async fn get_status(
    State(state): State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut conn = state
        .redis_client
        .get_async_connection()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let task_key = format!("task:{}", task_id);
    let exists: bool = conn
        .exists(&task_key)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !exists {
        return Err((StatusCode::NOT_FOUND, "Task ID not found".to_string()));
    }

    let data: std::collections::HashMap<String, String> = conn
        .hgetall(&task_key)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let status = data
        .get("status")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let result_raw = data.get("result").cloned();
    let error = data.get("error").cloned();
    let result = result_raw.and_then(|r| serde_json::from_str(&r).ok());

    Ok(Json(StatusResponse {
        task_id,
        status,
        result,
        error,
    }))
}

async fn health() -> impl IntoResponse {
    StatusCode::OK
}
