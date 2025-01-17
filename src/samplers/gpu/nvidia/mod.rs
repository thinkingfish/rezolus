use super::stats::*;
use super::*;
use crate::common::Nop;
use metriken::{DynBoxedMetric, MetricBuilder};
use nvml_wrapper::enum_wrappers::device::*;
use nvml_wrapper::Nvml;

const KB: i64 = 1024;
const MB: i64 = 1024 * KB;
const MHZ: i64 = 1_000_000;

#[distributed_slice(GPU_SAMPLERS)]
fn init(config: &Config) -> Box<dyn Sampler> {
    if let Ok(nvidia) = Nvidia::new(config) {
        Box::new(nvidia)
    } else {
        Box::new(Nop {})
    }
}

pub struct Nvidia {
    prev: Instant,
    next: Instant,
    interval: Duration,
    nvml: Nvml,
    pergpu_metrics: Vec<GpuMetrics>,
}

struct GpuMetrics {
    // current power usage in mW
    power_usage: DynBoxedMetric<metriken::Gauge>,

    // current die temperature in C
    temperature: DynBoxedMetric<metriken::Gauge>,

    // current pcie throughput in Bytes/s
    pcie_throughput_rx: DynBoxedMetric<metriken::Gauge>,
    pcie_throughput_tx: DynBoxedMetric<metriken::Gauge>,

    // current pcie bandwidth in Bytes/s
    pcie_bandwidth: DynBoxedMetric<metriken::Gauge>,

    // memory usage in bytes
    memory_free: DynBoxedMetric<metriken::Gauge>,
    memory_used: DynBoxedMetric<metriken::Gauge>,

    // current clock frequencies in Hz
    clock_graphics: DynBoxedMetric<metriken::Gauge>,
    clock_compute: DynBoxedMetric<metriken::Gauge>,
    clock_memory: DynBoxedMetric<metriken::Gauge>,
    clock_video: DynBoxedMetric<metriken::Gauge>,
}

impl Nvidia {
    pub fn new(_config: &Config) -> Result<Self, ()> {
        let now = Instant::now();
        let nvml = Nvml::init().map_err(|e| {
            error!("error initializing: {e}");
        })?;

        let devices = nvml
            .device_count()
            .map_err(|e| error!("error getting device count: {e}"))?;

        let mut pergpu_metrics = Vec::with_capacity(devices as _);

        for device in 0..devices {
            pergpu_metrics.push(GpuMetrics {
                power_usage: MetricBuilder::new("gpu/power/usage")
                    .metadata("id", format!("{}", device))
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                temperature: MetricBuilder::new("gpu/temperature")
                    .metadata("id", format!("{}", device))
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                pcie_throughput_rx: MetricBuilder::new("gpu/pcie/throughput")
                    .metadata("id", format!("{}", device))
                    .metadata("direction", "receive")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                pcie_throughput_tx: MetricBuilder::new("gpu/pcie/throughput")
                    .metadata("id", format!("{}", device))
                    .metadata("direction", "transmit")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                pcie_bandwidth: MetricBuilder::new("gpu/pcie/bandwidth")
                    .metadata("id", format!("{}", device))
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                memory_free: MetricBuilder::new("gpu/memory")
                    .metadata("id", format!("{}", device))
                    .metadata("state", "free")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                memory_used: MetricBuilder::new("gpu/memory")
                    .metadata("id", format!("{}", device))
                    .metadata("state", "used")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                clock_graphics: MetricBuilder::new("gpu/clock")
                    .metadata("id", format!("{}", device))
                    .metadata("type", "graphics")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                clock_compute: MetricBuilder::new("gpu/clock")
                    .metadata("id", format!("{}", device))
                    .metadata("type", "compute")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                clock_memory: MetricBuilder::new("gpu/clock")
                    .metadata("id", format!("{}", device))
                    .metadata("type", "memory")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
                clock_video: MetricBuilder::new("gpu/clock")
                    .metadata("id", format!("{}", device))
                    .metadata("type", "video")
                    .formatter(gpu_metric_formatter)
                    .build(metriken::Gauge::new()),
            });
        }

        Ok(Self {
            nvml,
            prev: now,
            next: now,
            interval: Duration::from_millis(50),
            pergpu_metrics,
        })
    }
}

impl Sampler for Nvidia {
    fn sample(&mut self) {
        let now = Instant::now();

        if now < self.next {
            return;
        }

        if let Err(e) = self.sample_nvml(now) {
            error!("error sampling: {e}");
        }

        // determine when to sample next
        let next = self.next + self.interval;

        // it's possible we fell behind
        if next > now {
            // if we didn't, sample at the next planned time
            self.next = next;
        } else {
            // if we did, sample after the interval has elapsed
            self.next = now + self.interval;
        }

        // mark when we last sampled
        self.prev = now;
    }
}

impl Nvidia {
    fn sample_nvml(&mut self, _now: Instant) -> Result<(), std::io::Error> {
        // current power usage in mW
        let mut power_usage = 0;

        // current pcie throughput scaled to Bytes/s
        let mut pcie_throughput_rx = 0;
        let mut pcie_throughput_tx = 0;

        // current pcie bandwidth scaled to Bytes/s
        let mut pcie_bandwidth = 0;

        // current memory stats in Bytes
        let mut gpu_memory_free = 0;
        let mut gpu_memory_used = 0;

        for (device_id, device_metrics) in self.pergpu_metrics.iter().enumerate() {
            if let Ok(device) = self.nvml.device_by_index(device_id as _) {
                /*
                 * power
                 */

                if let Ok(v) = device.power_usage() {
                    power_usage += v;
                    device_metrics.power_usage.set(v as _);
                }

                /*
                 * temperature
                 */

                if let Ok(v) = device.temperature(TemperatureSensor::Gpu) {
                    device_metrics.temperature.set(v as _);
                }

                /*
                 * pcie link
                 */

                if let Ok(v) = device
                    .pcie_throughput(PcieUtilCounter::Receive)
                    .map(|v| v as i64 * KB)
                {
                    pcie_throughput_rx += v;
                    device_metrics.pcie_throughput_rx.set(v);
                }

                if let Ok(v) = device
                    .pcie_throughput(PcieUtilCounter::Send)
                    .map(|v| v as i64 * KB)
                {
                    pcie_throughput_tx += v;
                    device_metrics.pcie_throughput_tx.set(v);
                }

                if let Ok(link_width) = device.current_pcie_link_width() {
                    if let Ok(link_gen) = device.current_pcie_link_gen() {
                        let v = match link_gen {
                            1 => 250 * MB,
                            2 => 500 * MB,
                            3 => 984 * MB,
                            4 => 1970 * MB,
                            5 => 3940 * MB,
                            6 => 7560 * MB,
                            7 => 15130 * MB,
                            _ => 0,
                        };

                        if v > 0 {
                            let v = v * link_width as i64;
                            pcie_bandwidth += v;
                            device_metrics.pcie_bandwidth.set(v as _);
                        }
                    }
                }

                /*
                 * memory
                 */

                if let Ok(memory_info) = device.memory_info() {
                    gpu_memory_free += memory_info.free;
                    gpu_memory_used += memory_info.used;
                    device_metrics.memory_free.set(memory_info.free as _);
                    device_metrics.memory_used.set(memory_info.used as _);
                }

                /*
                 * clocks
                 */

                if let Ok(frequency) = device.clock_info(Clock::Graphics).map(|f| f as i64 * MHZ) {
                    device_metrics.clock_graphics.set(frequency);
                }

                if let Ok(frequency) = device.clock_info(Clock::SM).map(|f| f as i64 * MHZ) {
                    device_metrics.clock_compute.set(frequency);
                }

                if let Ok(frequency) = device.clock_info(Clock::Memory).map(|f| f as i64 * MHZ) {
                    device_metrics.clock_memory.set(frequency);
                }

                if let Ok(frequency) = device.clock_info(Clock::Video).map(|f| f as i64 * MHZ) {
                    device_metrics.clock_video.set(frequency);
                }
            }
        }

        GPU_POWER_USAGE.set(power_usage as _);

        GPU_PCIE_BANDWIDTH.set(pcie_bandwidth as _);
        GPU_PCIE_THROUGHPUT_RX.set(pcie_throughput_rx as _);
        GPU_PCIE_THROUGHPUT_TX.set(pcie_throughput_tx as _);

        GPU_MEMORY_FREE.set(gpu_memory_free as _);
        GPU_MEMORY_USED.set(gpu_memory_used as _);

        Ok(())
    }
}
