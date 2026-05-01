//! Memory test for `metriken_query::Tsdb` — measures both the
//! load-time peak heap and the after-query peak heap, since both
//! contribute to what the WASM viewer's linear memory will grow to
//! and never shrink from.
//!
//! The streaming-only refactor in metriken-query 0.10.0 is meant to
//! reduce the after-query peak in particular; intermediate `Vec<f64>`
//! allocations from PromQL evaluation are what previously blew up the
//! WASM heap during dashboard render.
//!
//! Tracks heap allocations via a custom GlobalAlloc wrapper. Native
//! peak heap is the right proxy for WASM linear-memory growth: same
//! allocations, same high-water marks. (Native eventually frees on
//! drop; WASM keeps the pages forever.)

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use dashboard::ServiceExtension;
use metriken_query::{Bytes, QueryEngine, Tsdb};

struct TrackingAlloc;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding to the system allocator with the same layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let new = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(new, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAlloc = TrackingAlloc;

fn reset_peak_to_current() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn peak_since(baseline: usize) -> usize {
    PEAK.load(Ordering::Relaxed).saturating_sub(baseline)
}

fn current_since(baseline: usize) -> usize {
    CURRENT
        .load(Ordering::Relaxed)
        .saturating_sub(baseline)
}

fn data_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../site/viewer/data")
        .join(rel)
        .canonicalize()
        .unwrap_or_else(|e| panic!("test fixture {rel} unresolved: {e}"))
}

fn mb(bytes: usize) -> String {
    format!("{:>4} MiB", bytes / (1 << 20))
}

/// Load every service-template JSON from config/templates as
/// `ServiceExtension`. Skips category templates (which have a
/// different schema) and any file that fails to parse — the test
/// shouldn't be coupled to template-loading-error handling.
fn load_templates() -> Vec<ServiceExtension> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../config/templates")
        .canonicalize()
        .expect("config/templates unresolved");
    let mut exts = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read templates dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let body = std::fs::read_to_string(&path).expect("read template");
        let value: serde_json::Value = serde_json::from_str(&body).expect("parse template json");
        if value
            .get("category")
            .and_then(|b| b.as_bool())
            .unwrap_or(false)
        {
            continue;
        }
        match serde_json::from_value::<ServiceExtension>(value) {
            Ok(ext) => exts.push(ext),
            Err(e) => panic!("{} parse: {e}", path.display()),
        }
    }
    exts
}

/// Run every KPI query from every loaded service template against the
/// Tsdb. Misses (no matching metrics) are cheap — the engine returns
/// an empty result. Mirrors what the WASM viewer fires when rendering
/// service dashboards.
fn run_template_queries(tsdb: &Tsdb, templates: &[ServiceExtension]) -> usize {
    let engine = QueryEngine::new(tsdb);
    let (start, end) = engine.get_time_range();
    let mut count = 0usize;
    for ext in templates {
        for kpi in &ext.kpis {
            let q = kpi.effective_query();
            let _ = engine.query_range(&q, start, end, 1.0);
            count += 1;
        }
    }
    count
}

fn measure(path: &str, templates: &[ServiceExtension]) {
    let full = data_path(path);
    let bytes = std::fs::read(&full).unwrap_or_else(|e| panic!("read {full:?}: {e}"));

    // Anchor the measurement to the heap level just before the load —
    // strips out test-runner noise so the deltas are attributable to
    // metriken-query.
    reset_peak_to_current();
    let baseline = CURRENT.load(Ordering::Relaxed);

    // ── Load phase ─────────────────────────────────────────────────
    let load_started = std::time::Instant::now();
    let tsdb = Tsdb::load_from_bytes(Bytes::from(bytes))
        .unwrap_or_else(|e| panic!("load {full:?}: {e}"));
    let load_peak = peak_since(baseline);
    let load_resident = current_since(baseline);
    let load_ms = load_started.elapsed().as_millis();

    // Reset peak to current resident — we want the query phase's peak
    // to be measured incrementally over what loading already retained.
    reset_peak_to_current();
    let after_load = CURRENT.load(Ordering::Relaxed);

    // ── Query phase ────────────────────────────────────────────────
    let query_started = std::time::Instant::now();
    let n_queries = run_template_queries(&tsdb, templates);
    let query_peak = peak_since(after_load);
    let query_resident = current_since(after_load);
    let query_ms = query_started.elapsed().as_millis();

    // Total resident at end (load + queries combined).
    let combined_resident = current_since(baseline);
    let combined_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);

    drop(tsdb);

    eprintln!();
    eprintln!("── {} ────────────────────────────────────", path);
    eprintln!(
        "  load:        peak {}  resident {}  ({} ms)",
        mb(load_peak),
        mb(load_resident),
        load_ms,
    );
    eprintln!(
        "  queries:     peak {}  resident {}  ({} queries, {} ms)",
        mb(query_peak),
        mb(query_resident),
        n_queries,
        query_ms,
    );
    eprintln!(
        "  combined:    peak {}  resident {}    ← WASM linear-memory floor",
        mb(combined_peak),
        mb(combined_resident),
    );
}

#[test]
fn metriken_query_load_and_query_memory() {
    let templates = load_templates();
    eprintln!("loaded {} service templates from config/templates", templates.len());
    for path in [
        "demo.parquet",
        "cachecannon.parquet",
        "disagg/sglang-nixl-16c.parquet",
    ] {
        measure(path, &templates);
    }
}
