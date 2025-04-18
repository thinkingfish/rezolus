use super::*;

/// Runs Rezolus in `agent` mode in which it gathers systems telemetry and
/// exposes metrics on an OTel/Prometheus compatible endpoint and a
/// Rezolus-specific msgpack endpoint.
///
/// This is the default mode for running Rezolus.
pub fn run(args: AgentArgs) {
    // load config from file
    let config: Arc<Config> = {
        let file = args.config;
        debug!("loading config: {:?}", file);
        match Config::load(&file) {
            Ok(c) => c.into(),
            Err(error) => {
                eprintln!("error loading config file: {:?}\n{error}", file);
                std::process::exit(1);
            }
        }
    };

    // configure debug log
    let debug_output: Box<dyn Output> = Box::new(Stderr::new());

    let level = config.log().level();

    let debug_log = if level <= Level::Info {
        LogBuilder::new().format(ringlog::default_format)
    } else {
        LogBuilder::new()
    }
    .output(debug_output)
    .build()
    .expect("failed to initialize debug log");

    let mut log = MultiLogBuilder::new()
        .level_filter(level.to_level_filter())
        .default(debug_log)
        .build()
        .start();

    // initialize async runtime
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .thread_name("rezolus")
        .build()
        .expect("failed to launch async runtime");

    // spawn logging thread
    rt.spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let _ = log.flush();
        }
    });

    let mut samplers = Vec::new();

    for init in SAMPLERS {
        if let Ok(Some(s)) = init(config.clone()) {
            samplers.push(s);
        }
    }

    let samplers = Arc::new(samplers.into_boxed_slice());

    rt.spawn(async move {
        exposition::http::serve(config, samplers).await;
    });

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
