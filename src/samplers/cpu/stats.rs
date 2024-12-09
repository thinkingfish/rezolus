use metriken::*;

#[metric(
    name = "cpu/cores",
    description = "The total number of logical cores that are currently online"
)]
pub static CPU_CORES: LazyGauge = LazyGauge::new(Gauge::default);

#[metric(
    name = "cpu/usage/total",
    description = "The amount of CPU time spent busy",
    formatter = cpu_usage_total_formatter,
    metadata = { state = "busy", unit = "nanoseconds" }
)]
pub static CPU_USAGE_BUSY: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "cpu/usage/total",
    description = "The amount of CPU time spent executing normal tasks is user mode",
    formatter = cpu_usage_total_formatter,
    metadata = { state = "user", unit = "nanoseconds" }
)]
pub static CPU_USAGE_USER: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "cpu/usage/total",
    description = "The amount of CPU time spent executing low priority tasks in user mode",
    formatter = cpu_usage_total_formatter,
    metadata = { state = "nice", unit = "nanoseconds" }
)]
pub static CPU_USAGE_NICE: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "cpu/usage/total",
    description = "The amount of CPU time spent executing tasks in kernel mode",
    formatter = cpu_usage_total_formatter,
    metadata = { state = "system", unit = "nanoseconds" }
)]
pub static CPU_USAGE_SYSTEM: LazyCounter = LazyCounter::new(Counter::default);

pub fn cpu_usage_total_formatter(metric: &MetricEntry, format: Format) -> String {
    match format {
        Format::Simple => {
            let state = metric
                .metadata()
                .get("state")
                .expect("no `state` for metric formatter");
            format!("cpu/usage/{state}/total")
        }
        _ => metric.name().to_string(),
    }
}
