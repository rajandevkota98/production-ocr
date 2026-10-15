# Stage 1: Cargo Chef Planner
FROM rust:1.88-alpine as chef
RUN apk add --no-cache musl-dev
WORKDIR /app
RUN cargo install cargo-chef --locked

# Stage 2: Prepare Dependency Recipe
FROM chef as planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Cook Dependencies and Build Release Binary
FROM chef as builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --release

# Stage 4: Minimal Final Runtime Image
FROM alpine:3.20
RUN apk add --no-cache libgcc
WORKDIR /app
COPY --from=builder /app/target/release/production-ocr-gateway-rust /app/gateway

ENV PORT=7860
EXPOSE 7860

CMD ["/app/gateway"]
