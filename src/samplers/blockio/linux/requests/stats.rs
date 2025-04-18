use crate::common::HISTOGRAM_GROUPING_POWER;
use metriken::*;

#[metric(
    name = "blockio_size",
    description = "Distribution of blockio operation sizes in bytes",
    metadata = { unit = "bytes" }
)]
pub static BLOCKIO_SIZE: RwLockHistogram = RwLockHistogram::new(HISTOGRAM_GROUPING_POWER, 64);

#[metric(
    name = "blockio_read_size",
    description = "Distribution of blockio read operation sizes in bytes",
    metadata = { unit = "bytes" }
)]
pub static BLOCKIO_READ_SIZE: RwLockHistogram = RwLockHistogram::new(HISTOGRAM_GROUPING_POWER, 64);

#[metric(
    name = "blockio_write_size",
    description = "Distribution of blockio write operation sizes in bytes",
    metadata = { unit = "bytes" }
)]
pub static BLOCKIO_WRITE_SIZE: RwLockHistogram = RwLockHistogram::new(HISTOGRAM_GROUPING_POWER, 64);

#[metric(
    name = "blockio_operations",
    description = "The number of completed read operations for block devices",
    metadata = { op = "read", unit = "operations" }
)]
pub static BLOCKIO_READ_OPS: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_operations",
    description = "The number of completed write operations for block devices",
    metadata = { op = "write", unit = "operations" }
)]
pub static BLOCKIO_WRITE_OPS: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_operations",
    description = "The number of completed discard operations for block devices",
    metadata = { op = "discard", unit = "operations" }
)]
pub static BLOCKIO_DISCARD_OPS: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_operations",
    description = "The number of completed flush operations for block devices",
    metadata = { op = "flush", unit = "operations" }
)]
pub static BLOCKIO_FLUSH_OPS: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_bytes",
    description = "The number of bytes read for block devices",
    metadata = { op = "read", unit = "bytes" }
)]
pub static BLOCKIO_READ_BYTES: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_bytes",
    description = "The number of bytes written for block devices",
    metadata = { op = "write", unit = "bytes" }
)]
pub static BLOCKIO_WRITE_BYTES: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_bytes",
    description = "The number of bytes discarded for block devices",
    metadata = { op = "discard", unit = "bytes" }
)]
pub static BLOCKIO_DISCARD_BYTES: LazyCounter = LazyCounter::new(Counter::default);

#[metric(
    name = "blockio_bytes",
    description = "The number of bytes flushed for block devices",
    metadata = { op = "flush", unit = "bytes" }
)]
pub static BLOCKIO_FLUSH_BYTES: LazyCounter = LazyCounter::new(Counter::default);
